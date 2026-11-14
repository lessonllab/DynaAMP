"""Use MuJoCo observations with the shared LoopMimic inference module."""

from pathlib import Path

from shared.dance import DancePolicy, Observation, TargetPublisher, rotation


class Dance(DancePolicy):
    def __init__(self, bundle: Path, motion: Path, env):
        super().__init__(bundle, motion)
        model = env.mj_model
        if self.meta["joint_names"] != [model.joint(int(i)).name for i in env.body_joint_index]:
            raise ValueError("SONIC DDS body-motor order does not match the LoopMimic export")
        self.data = env.mj_data
        self.q_ids = model.jnt_qposadr[env.body_joint_index]
        self.v_ids = model.jnt_dofadr[env.body_joint_index]
        self.publisher = TargetPublisher(self.meta)

    def observation(self) -> Observation:
        d = self.data
        return Observation(d.qpos[:3], d.qpos[3:7], d.qpos[self.q_ids], d.qvel[:3],
                           rotation(d.qpos[3:7]) @ d.qvel[3:6], d.qvel[self.v_ids])

    def start(self):
        super().start(self.observation())

    def step(self):
        self.publisher.send(0, self.target(self.observation()))

    def stop(self):
        self.publisher.send(0, None)

    def close(self):
        self.publisher.close()
