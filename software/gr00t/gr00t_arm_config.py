# GR00T modality config for the SG90 gr00t-arm (one front camera, 4 arm
# joints + gripper). Pass it to Isaac-GR00T with
#   --embodiment-tag NEW_EMBODIMENT --modality-config-path <this file>
# Keys must match meta/modality.json written by software/export_dataset.py.
# Based on examples/SO100/so100_config.py from NVIDIA Isaac-GR00T (Apache-2.0).

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

gr00t_arm_config = {
    # the phone camera on the mast; current frame only
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front"]),
    # servo angles (degrees) the arm was last commanded
    "state": ModalityConfig(delta_indices=[0], modality_keys=["single_arm", "gripper"]),
    # GR00T predicts the next 16 commands (~0.5 s at 30 fps) in one go
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            # arm joints as changes from the current angles: generalises better
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF,
                         format=ActionFormat.DEFAULT),
            # gripper as an absolute target: it's basically open / closed
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                         format=ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig(delta_indices=[0],
                               modality_keys=["annotation.human.task_description"]),
}

register_modality_config(gr00t_arm_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
