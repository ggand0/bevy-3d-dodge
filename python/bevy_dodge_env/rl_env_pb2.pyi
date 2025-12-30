from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ObservationSpace(_message.Message):
    __slots__ = ("shape", "dtype", "low", "high")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    LOW_FIELD_NUMBER: _ClassVar[int]
    HIGH_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    low: float
    high: float
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., low: _Optional[float] = ..., high: _Optional[float] = ...) -> None: ...

class ActionSpace(_message.Message):
    __slots__ = ("discrete", "box")
    DISCRETE_FIELD_NUMBER: _ClassVar[int]
    BOX_FIELD_NUMBER: _ClassVar[int]
    discrete: DiscreteSpace
    box: BoxSpace
    def __init__(self, discrete: _Optional[_Union[DiscreteSpace, _Mapping]] = ..., box: _Optional[_Union[BoxSpace, _Mapping]] = ...) -> None: ...

class DiscreteSpace(_message.Message):
    __slots__ = ("n",)
    N_FIELD_NUMBER: _ClassVar[int]
    n: int
    def __init__(self, n: _Optional[int] = ...) -> None: ...

class BoxSpace(_message.Message):
    __slots__ = ("shape", "low", "high")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    LOW_FIELD_NUMBER: _ClassVar[int]
    HIGH_FIELD_NUMBER: _ClassVar[int]
    shape: _containers.RepeatedScalarFieldContainer[int]
    low: float
    high: float
    def __init__(self, shape: _Optional[_Iterable[int]] = ..., low: _Optional[float] = ..., high: _Optional[float] = ...) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResetResponse(_message.Message):
    __slots__ = ("observation", "image_observation", "info")
    class InfoEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    IMAGE_OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    observation: _containers.RepeatedScalarFieldContainer[float]
    image_observation: bytes
    info: _containers.ScalarMap[str, str]
    def __init__(self, observation: _Optional[_Iterable[float]] = ..., image_observation: _Optional[bytes] = ..., info: _Optional[_Mapping[str, str]] = ...) -> None: ...

class StepRequest(_message.Message):
    __slots__ = ("discrete_action", "continuous_action")
    DISCRETE_ACTION_FIELD_NUMBER: _ClassVar[int]
    CONTINUOUS_ACTION_FIELD_NUMBER: _ClassVar[int]
    discrete_action: int
    continuous_action: ContinuousAction
    def __init__(self, discrete_action: _Optional[int] = ..., continuous_action: _Optional[_Union[ContinuousAction, _Mapping]] = ...) -> None: ...

class ContinuousAction(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...

class StepResponse(_message.Message):
    __slots__ = ("observation", "image_observation", "reward", "done", "truncated", "info")
    class InfoEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    IMAGE_OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    REWARD_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    observation: _containers.RepeatedScalarFieldContainer[float]
    image_observation: bytes
    reward: float
    done: bool
    truncated: bool
    info: _containers.ScalarMap[str, str]
    def __init__(self, observation: _Optional[_Iterable[float]] = ..., image_observation: _Optional[bytes] = ..., reward: _Optional[float] = ..., done: bool = ..., truncated: bool = ..., info: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ConfigureRequest(_message.Message):
    __slots__ = ("level", "action_space_type", "sprint_multiplier", "spawn_angle_degrees", "observation_mode", "thrower_delay_seconds", "image_obs_width", "image_obs_height", "image_grayscale", "collision_penalty", "survival_reward", "dodge_bonus_threshold", "dodge_bonus_multiplier", "projectile_speed", "projectile_spawn_interval", "max_projectiles", "player_speed")
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    ACTION_SPACE_TYPE_FIELD_NUMBER: _ClassVar[int]
    SPRINT_MULTIPLIER_FIELD_NUMBER: _ClassVar[int]
    SPAWN_ANGLE_DEGREES_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_MODE_FIELD_NUMBER: _ClassVar[int]
    THROWER_DELAY_SECONDS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_OBS_WIDTH_FIELD_NUMBER: _ClassVar[int]
    IMAGE_OBS_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_GRAYSCALE_FIELD_NUMBER: _ClassVar[int]
    COLLISION_PENALTY_FIELD_NUMBER: _ClassVar[int]
    SURVIVAL_REWARD_FIELD_NUMBER: _ClassVar[int]
    DODGE_BONUS_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    DODGE_BONUS_MULTIPLIER_FIELD_NUMBER: _ClassVar[int]
    PROJECTILE_SPEED_FIELD_NUMBER: _ClassVar[int]
    PROJECTILE_SPAWN_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    MAX_PROJECTILES_FIELD_NUMBER: _ClassVar[int]
    PLAYER_SPEED_FIELD_NUMBER: _ClassVar[int]
    level: int
    action_space_type: str
    sprint_multiplier: float
    spawn_angle_degrees: float
    observation_mode: str
    thrower_delay_seconds: float
    image_obs_width: int
    image_obs_height: int
    image_grayscale: bool
    collision_penalty: float
    survival_reward: float
    dodge_bonus_threshold: float
    dodge_bonus_multiplier: float
    projectile_speed: float
    projectile_spawn_interval: float
    max_projectiles: int
    player_speed: float
    def __init__(self, level: _Optional[int] = ..., action_space_type: _Optional[str] = ..., sprint_multiplier: _Optional[float] = ..., spawn_angle_degrees: _Optional[float] = ..., observation_mode: _Optional[str] = ..., thrower_delay_seconds: _Optional[float] = ..., image_obs_width: _Optional[int] = ..., image_obs_height: _Optional[int] = ..., image_grayscale: bool = ..., collision_penalty: _Optional[float] = ..., survival_reward: _Optional[float] = ..., dodge_bonus_threshold: _Optional[float] = ..., dodge_bonus_multiplier: _Optional[float] = ..., projectile_speed: _Optional[float] = ..., projectile_spawn_interval: _Optional[float] = ..., max_projectiles: _Optional[int] = ..., player_speed: _Optional[float] = ...) -> None: ...

class ConfigureResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SetLevelRequest(_message.Message):
    __slots__ = ("level",)
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    level: int
    def __init__(self, level: _Optional[int] = ...) -> None: ...

class SetLevelResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StartTrainingRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StartTrainingResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class EndTrainingRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class EndTrainingResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ObservationSpaceRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ActionSpaceRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
