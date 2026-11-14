"""The upstream 13-body AMP feature convention, computed from GenCoord state.

All features are invariant to a global rigid transform. Root-frame FK therefore
uses identity root orientation and root-frame velocities, preserving exact input
derivatives without reconstructing yaw or extracting roll/pitch angles.
"""
import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse, quat_conjugate, quat_mul

AMP_BODIES = (
    'pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link',
    'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link',
    'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_yaw_link',
    'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link',
)


def encode_bodies(positions, rotations, velocities, angular, bodies, anchor):
    q = rotations[:, bodies]
    anchor_q = rotations[:, anchor:anchor + 1].expand_as(q)
    relative = quat_apply_inverse(anchor_q, positions[:, bodies] - positions[:, anchor:anchor + 1])
    orientation = matrix_from_quat(quat_mul(quat_conjugate(anchor_q), q))[..., :2]
    return torch.cat((relative.flatten(1), orientation.flatten(1),
                      quat_apply_inverse(q, velocities[:, bodies]).flatten(1),
                      quat_apply_inverse(q, angular[:, bodies]).flatten(1)), -1)


class BodyAMPFeatures:
    def __init__(self, motion, body_names):
        self.fk = motion.fk
        names = list(body_names)
        self.bodies = [names.index(n) for n in AMP_BODIES]
        self.anchor = names.index('torso_link')
        self.forward = torch.compile(self._forward, fullgraph=True, dynamic=True) if motion.joint_pos.is_cuda else self._forward

    def __call__(self, states):
        return self.forward(states)

    @torch.no_grad()
    def expert_frames(self, states, chunk_size=8192):
        """Bound preprocessing memory without changing any training batch."""
        return torch.cat([self(part) for part in states.split(chunk_size)], 0)

    def metadata(self):
        """Immutable FK constants for self-contained paired-feedback replay."""
        return {"bodies": self.bodies, "anchor": self.anchor,
                "grounded": isinstance(self, GroundedBodyAMPFeatures),
                **{key: getattr(self.fk, key).detach().cpu().clone()
                   for key in ("consts", "col", "qpos0")}}

    @classmethod
    def from_metadata(cls, state, device):
        from mjlab.utils.forward_kinematics import ForwardKinematics
        kind = GroundedBodyAMPFeatures if state.get("grounded", False) else cls
        result = kind.__new__(kind)
        result.fk = ForwardKinematics.__new__(ForwardKinematics)
        for key in ("consts", "col", "qpos0"):
            setattr(result.fk, key, state[key].to(device))
        result.bodies, result.anchor = state["bodies"], state["anchor"]
        result.forward = torch.compile(result._forward, fullgraph=True, dynamic=True) if torch.device(device).type == "cuda" else result._forward
        return result

    def _forward(self, states):
        shape = states.shape[:-1]
        flat = states.reshape(-1, states.shape[-1])
        joints = (flat.shape[-1] - 10) // 2
        position = torch.zeros_like(flat[:, 4:7])
        rotation = torch.cat((torch.ones_like(flat[:, :1]), torch.zeros_like(position)), -1)
        values = self.fk(position, rotation, flat[:, 10:10 + joints],
                         flat[:, 4:7], flat[:, 7:10], flat[:, 10 + joints:])
        return encode_bodies(*values, self.bodies, self.anchor).reshape(*shape, 195)


class GroundedBodyAMPFeatures(BodyAMPFeatures):
    """Body195 plus root height and projected gravity, without reference leakage.

    Keep horizontal-translation and heading invariance while exposing ground
    height and tilt. Reference, simulation and imagination use identical code.
    """

    def _forward(self, states):
        return torch.cat((super()._forward(states), states[..., :4]), dim=-1)
