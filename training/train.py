"""Train or resume LoopMimic v2, with evaluations at exact update counts."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import imageio_ffmpeg
from mjlab.rl import RslRlVecEnvWrapper
from supermimic.envs.transition_capture import TransitionCaptureEnv
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import SuperMimicRunner
from supermimic.scripts.compare_amp import TASK, json_value
from supermimic.scripts.compare_wmrl_branch import branch_env_cfg
from supermimic.scripts.train_utils import setup_device

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/lafan1_g1_50hz'
PROTOCOL = ROOT / 'config/evaluation_protocol.json'
CONFIG = ROOT / 'config/runner_014000.json'
CLIPS = ['walk1_subject1', 'run1_subject5', 'dance1_subject2', 'jumps1_subject1']


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def validate_video(path, expected_seconds):
    # Use the ffmpeg binary packaged with imageio; do not depend on system ffprobe.
    frames = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    metadata = next(frames)
    count = sum(1 for _ in frames)  # Decode every frame, not just container headers.
    if count < round(expected_seconds * metadata['fps']) or metadata['size'][0] <= 0:
        raise RuntimeError(f'Incomplete video: {path}, frames={count}, metadata={metadata}')
    return dict(frames=count, fps=metadata['fps'], seconds=count/metadata['fps'], size=metadata['size'])


def main(output: Path, target=14000, every=1000, smoke=False, resume: Path | None = None,
         stop_on_full_coverage: bool = False, data: Path = DATA, skip_videos: bool = False,
         verify_reference_init: bool = False):
    output = output.resolve()
    data = data.resolve()
    os.environ.setdefault('WANDB_MODE', 'offline')
    os.environ.setdefault('WANDB_DIR', str(output))
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'status.json'
    if resume is None and status_path.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if target <= 0 or every <= 0 or (not smoke and target % every):
        raise ValueError('Use a positive target aligned to the evaluation interval')
    settings = json.loads(CONFIG.read_text())
    num_envs = settings['num_envs']
    reference = json.loads(PROTOCOL.read_text())
    manifest_path = data / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'{manifest_path}. Run prepare_data.py first.')
    manifest = json.loads(manifest_path.read_text())
    entries = {Path(x['output']).name: x for x in manifest}
    files = sorted(data.glob('*.npz'))
    if ({file.stem for file in files} != set(reference['target_steps'])
            or len(files) != len(manifest) or set(entries) != {file.name for file in files}):
        raise ValueError('Use the 40 G1 trajectories from the release manifest')
    for file in files:
        if hashlib.sha256(file.read_bytes()).hexdigest() != entries[file.name]['output_sha256']:
            raise ValueError(f'Trajectory hash mismatch: {file}')
    data_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    started = time.time()
    rows = []
    start_updates = 0
    previous_training_s = 0.0
    if resume is None:
        cfg = load_runner_cfg(settings)
        cfg.run_name = output.name
    else:
        resume = resume.resolve()
        origin = resume.parent
        origin_status_path = origin / 'status.json'
        if not origin_status_path.exists():
            raise ValueError('Continuation requires its original run directory')
        previous_status = json.loads(origin_status_path.read_text())
        if origin != output and (status_path.exists() or previous_status['phase'] != 'complete'):
            raise ValueError('Continuing a completed run in another directory requires a new destination')
        started = previous_status['started_unix']
        prior_evaluations = origin / 'evaluations.json'
        if prior_evaluations.exists():
            rows = json.loads(prior_evaluations.read_text())
        state = torch.load(resume, map_location='cpu', weights_only=False)
        start_updates = state['policy_updates']
        previous_training_s = state['elapsed_training_s']
        cfg = load_runner_cfg(state)
        assert int(state['amp']['updates']) == start_updates
        del state
        if start_updates >= target:
            raise ValueError('The resume checkpoint must precede the target')
        if any(row['updates'] > start_updates for row in rows):
            raise ValueError('The resume checkpoint must include all retained evaluations')
        if smoke != json.loads((origin / 'contract.json').read_text())['smoke']:
            raise ValueError('Use the same smoke or production mode when resuming')
        cfg.resume_checkpoint = str(resume)
        if origin != output:
            for name in ('contract.json', 'env_config.json', 'amp_gradient_audit.json'):
                if (origin / name).exists():
                    shutil.copy2(origin / name, output / name)
            write(output / 'evaluations.json', rows)
    completed = tuple(row['updates'] for row in rows)
    pending = (target,) if smoke else tuple(
        range((start_updates // every + 1) * every, target + 1, every)
    )
    evaluate_resume = (resume is not None and start_updates not in completed
                       and start_updates > 0 and start_updates % every == 0)
    milestones = completed + ((start_updates,) if evaluate_resume else ()) + pending

    def status(phase, **details):
        write(status_path, dict(phase=phase, pid=os.getpid(), started_unix=started,
            updated_unix=time.time(), from_scratch=True, start_policy_updates=0,
            resume_policy_updates=start_updates, resume_checkpoint=str(resume) if resume else None,
            target_policy_updates=target, evaluation_updates=milestones,
            stop_on_full_coverage=stop_on_full_coverage,
            evaluated=rows, log=str(output / 'train.log'), **details))
    def child(command, log, **details):
        with log.open('w') as stream, subprocess.Popen(command, stdout=stream,
                stderr=subprocess.STDOUT, cwd=ROOT) as proc:
            status('evaluation', child_pid=proc.pid, child_log=str(log), **details)
            if proc.wait():
                raise RuntimeError(f'Evaluation/recording failed; see {log}')
    def videos(updates, checkpoint):
        outputs = []
        for clip in (CLIPS[1:2] if smoke else CLIPS):
            path = output / 'videos' / f'{updates:06d}_{clip}.mp4'
            path.parent.mkdir(exist_ok=True)
            child([sys.executable, '-u', '-m', 'supermimic.scripts.evaluate_matched_amp', 'video',
                   '--checkpoint', str(checkpoint), '--output', str(path), '--clip', clip,
                   '--data', str(data), '--seed', str(reference['seed']),
                   '--seconds', '2' if smoke else '20'], path.with_suffix('.log'), updates=updates)
            write(path.with_suffix('.validation.json'), validate_video(path, 2 if smoke else 20))
            outputs.append(str(path))
        return outputs
    def evaluate(updates, checkpoint):
        before = time.monotonic()
        path = output / f'evaluation_{updates:06d}.json'
        if not path.exists():
            log = path.with_suffix('.log')
            if log.exists():
                log = path.with_suffix('.continuation.log')
            child([sys.executable, '-u', '-m', 'supermimic.scripts.evaluate_matched_amp', 'evaluate',
                   '--checkpoint', str(checkpoint), '--output', str(path), '--data', str(data),
                   '--replicas', str(reference['replicas_per_clip']),
                   '--seed', str(reference['seed'])], log, updates=updates, checkpoint=str(checkpoint))
        result = json.loads(path.read_text())
        for key in ('seed', 'replicas_per_clip', 'clips'):
            if result[key] != reference[key]:
                raise ValueError(f'Evaluation protocol changed: {key}')
        expected = [(clip, replica, steps) for clip, steps in reference['target_steps'].items()
                    for replica in range(reference['replicas_per_clip'])]
        if [(e['clip'], e['replica'], e['target_steps']) for e in result['episodes']] != expected:
            raise ValueError('Evaluation clip list changed')
        if verify_reference_init:
            for key in ('initial_physics_state_sha256', 'sim_context_sha256'):
                if result[key] != reference[key]:
                    raise ValueError(f'Evaluation initial state differs from the release: {key}')
        if result['checkpoint'] != str(checkpoint):
            raise ValueError('Evaluation checkpoint mismatch')
        missed = [e for e in result['episodes'] if not e['completed']]
        row = dict(updates=updates, completed=len(result['episodes'])-len(missed), episodes=len(result['episodes']),
                   training_real_sim_transitions=result['training_real_sim_transitions'], summary=result['summary'],
                   checkpoint=str(checkpoint), evaluation=str(path), missed=missed,
                   evaluation_seconds=time.monotonic()-before, policy='ema')
        first_full = not missed and not any(r['completed'] == r['episodes'] for r in rows)
        stop = stop_on_full_coverage and not missed
        if not skip_videos and (smoke or updates in (milestones[0], target) or first_full):
            row['videos'] = videos(updates, checkpoint)
        rows.append(row)
        write(output / 'evaluations.json', rows)
        print(f'[LOOPMIMIC_V2] update={updates} completed={row["completed"]}/{row["episodes"]}', flush=True)
        status('finalizing' if stop else 'training', last_evaluation_updates=updates)
        return stop

    status('initializing')
    try:
        setup_device('cuda:0')
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        cfg.max_amp_updates = target
        if smoke:
            cfg.amp_feedback.interval = 20
            cfg.amp_feedback.min_amp_updates = 4
        assert cfg.policy.batch_size == 4096 and cfg.policy.horizon == 32 and cfg.policy.num_batches == 1
        assert cfg.wm.batch_size == 2048 and cfg.wm.num_batches == 10 and cfg.wm.horizon == 8
        assert cfg.wandb_run_path is None
        ecfg = branch_env_cfg(num_envs, 0, data=data)
        ecfg.commands['motion'].sampling_mode = 'external'
        env_record = asdict(ecfg)
        env_record['commands']['motion']['motion_file'] = '*.npz'
        env_json = json.dumps(env_record, sort_keys=True, default=json_value)
        import supermimic.runners.supermimic_runner as owner
        source = Path(owner.__file__).resolve().parents[2]
        contract_path = output / 'contract.json'
        env_hash = hashlib.sha256(env_json.encode()).hexdigest()
        if resume is None:
            (output / 'env_config.json').write_text(env_json)
            write(contract_path, dict(
                method='loopmimic_v2', from_scratch=True, seed=0, num_envs=num_envs,
                target_policy_updates=target, evaluation_updates=milestones, smoke=smoke,
                dataset=str(data), dataset_manifest_sha256=data_hash,
                env_cfg_sha256=env_hash, runner_cfg=asdict(cfg),
                loaded_checkpoints=[], source_root=str(source),
                sources={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in source.rglob('*.py')},
                evaluation_reference=str(PROTOCOL), evaluation_policy='ema',
                interaction_unit='All real simulation transitions, including all feedback/repeat/holdout branches. Evaluation counted separately.',
                changes=['Robot-frame root position error into unchanged-width decoder',
                    'Ground-aware 199D AMP features', 'EMA evaluation policy, live training policy',
                    'Return/failure reset sampling', 'Real terminal targets and masked prefix WM windows',
                    'Paired AMP response calibration of WM', 'Feedback replay shared by WM, policy starts and discriminator']))
        else:
            contract = json.loads(contract_path.read_text())
            if env_hash != contract['env_cfg_sha256'] or data_hash != contract['dataset_manifest_sha256']:
                raise ValueError('Continuation changed the training environment or data')
            write(output / 'continuation.json', dict(checkpoint=str(resume), start_policy_updates=start_updates,
                target_policy_updates=target, evaluation_updates=milestones, evaluation_interval=every,
                prior_training_seconds=previous_training_s, source_root=str(source), runner_cfg=asdict(cfg)))
        stop_at_resume = False
        if stop_on_full_coverage and resume is not None and rows and rows[-1]['updates'] == start_updates:
            stop_at_resume = rows[-1]['completed'] == rows[-1]['episodes']
        if evaluate_resume:
            stop_at_resume = evaluate(start_updates, resume)
        elif resume is not None:
            write(output / 'evaluations.json', rows)
        if stop_at_resume:
            status('finalizing', last_evaluation_updates=start_updates)
        else:
            base = TransitionCaptureEnv(ecfg, device='cuda:0')
            env = RslRlVecEnvWrapper(base)
            runner = build_runner(env, cfg, output, resume)
            status('training')
            runner.run(update_checkpoints=pending, evaluate_checkpoint=evaluate)
        if stop_at_resume:
            progress = dict(policy_updates=start_updates, amp_updates=start_updates,
                            elapsed_training_s=previous_training_s,
                            real_sim_transitions=rows[-1]['training_real_sim_transitions'])
            if not (output / 'progress.json').is_file():
                write(output / 'progress.json', progress)
        else:
            progress = json.loads((output/'progress.json').read_text())
        final_updates = progress['policy_updates']
        evaluated = tuple(r['updates'] for r in rows)
        stopped_on_full = (stop_on_full_coverage and bool(rows)
                           and rows[-1]['completed'] == rows[-1]['episodes'])
        if (progress['amp_updates'] != final_updates or not evaluated or evaluated[-1] != final_updates
                or evaluated != milestones[:len(evaluated)] or (final_updates != target and not stopped_on_full)):
            raise RuntimeError('Requested update/evaluation schedule incomplete')
        if smoke and target >= 24:
            records = [json.loads(x) for x in (output/'amp_feedback_rounds.jsonl').read_text().splitlines()]
            assert len(records) >= 2 and all(r['priority_windows_stored'] > 0 for r in records)
        full = [r for r in rows if r['completed'] == r['episodes']]
        stop_reason = 'full_coverage' if stopped_on_full else 'target_updates'
        result = dict(from_scratch=True, final_updates=final_updates, stop_reason=stop_reason,
                      max_policy_updates=target, stop_on_full_coverage=stop_on_full_coverage,
                      elapsed_training_s=progress['elapsed_training_s'],
                      first_full_coverage_updates=full[0]['updates'] if full else None,
                      evaluated=rows, real_sim_transitions=progress['real_sim_transitions'])
        write(output/'results.json', result)
        status('complete', results=str(output/'results.json'), final_updates=final_updates, stop_reason=stop_reason,
               first_full_coverage_updates=result['first_full_coverage_updates'])
    except Exception as exc:
        status('failed', error=str(exc))
        raise


def build_runner(env, cfg, output, resume):
    class Runner(SuperMimicRunner):
        def _create_log_dir(self, *args):
            return output

        def _step_and_store(self, actions, curr_states, actor_td=None, extra_fields=None):
            result = super()._step_and_store(actions, curr_states, actor_td, extra_fields)
            if resume is None and not hasattr(self, '_transition_verified'):
                live = self.env.unwrapped.physics_state()
                done = result[2].bool()
                target = self.env.unwrapped.transition_next_state
                error = (target-live).abs().amax(-1)
                record = dict(nonterminal_error=float(error[~done].max()) if (~done).any() else 0.0,
                    observation_error=float((target[~done]-result[0]['full_state'][~done]).abs().max()) if (~done).any() else 0.0,
                    terminal_rows=int(done.sum()), terminal_targets_distinct=int((error[done]>1e-5).sum()))
                assert record['nonterminal_error'] < 1e-5
                assert record['observation_error'] < 1e-4
                write(output/'transition_audit.json', record)
                self._transition_verified = True
            return result

    runner = Runner(env, cfg, TASK, device='cuda:0')
    if resume is None:
        runner._save_training_state(output, 0, snapshot=output/'initial_random.pt')
    else:
        audit = output / 'amp_gradient_audit.json'
        runner._amp_audited = audit.is_file() and json.loads(audit.read_text())['passed']
    return runner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=int, default=14000)
    parser.add_argument('--every', type=int, default=1000)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-on-full-coverage', action='store_true')
    parser.add_argument('--skip-videos', action='store_true')
    parser.add_argument('--verify-reference-init', action='store_true',
                        help='Require exact release evaluation state hashes')
    main(**vars(parser.parse_args()))
