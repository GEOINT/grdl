# -*- coding: utf-8 -*-
"""
Tunable Parameter Annotations - Declarative parameter constraints via typing.Annotated.

Provides constraint marker types (``Range``, ``Options``, ``Desc``) for use
inside ``typing.Annotated`` annotations on ``ImageProcessor`` subclasses, plus
the ``ParamSpec`` introspection class and collection/init-generation utilities
consumed by ``ImageProcessor.__init_subclass__``.

Usage
-----
Declare tunable parameters as class-body annotations::

    from typing import Annotated
    from grdl.image_processing.params import Range, Options, Desc

    class MyFilter(ImageTransform):
        sigma: Annotated[float, Range(min=0.1, max=100.0), Desc('Gaussian sigma')] = 2.0
        method: Annotated[str, Options('bilinear', 'nearest'), Desc('Interpolation')] = 'bilinear'

Parameters are automatically collected into ``cls.__param_specs__`` at class
definition time. An ``__init__`` is auto-generated unless the class defines
its own.

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-10

Modified
--------
2026-02-10
"""

# Standard library
import inspect
from typing import (
    Annotated,
    Any,
    Dict,
    Optional,
    Tuple,
    Union,
    get_origin,
    get_type_hints,
)


# =====================================================================
# Constraint marker types  (used inside Annotated[...])
# =====================================================================

class ParamMeta:
    """Base marker for tunable parameter metadata in ``Annotated`` types.

    Any ``Annotated`` class-body field whose metadata includes at least one
    ``ParamMeta`` subclass instance is treated as a tunable parameter by
    ``ImageProcessor.__init_subclass__``.
    """


class Range(ParamMeta):
    """Inclusive numeric range constraint.

    Parameters
    ----------
    min : int or float, optional
        Minimum allowed value (inclusive).
    max : int or float, optional
        Maximum allowed value (inclusive).
    """

    __slots__ = ('min', 'max')

    def __init__(
        self,
        min: Optional[Union[int, float]] = None,
        max: Optional[Union[int, float]] = None,
    ) -> None:
        self.min = min
        self.max = max

    def __repr__(self) -> str:
        parts = []
        if self.min is not None:
            parts.append(f"min={self.min!r}")
        if self.max is not None:
            parts.append(f"max={self.max!r}")
        return f"Range({', '.join(parts)})"


class Options(ParamMeta):
    """Discrete choice constraint.

    Parameters
    ----------
    *choices
        Allowed values.  Must supply at least one.
    """

    __slots__ = ('choices',)

    def __init__(self, *choices: Any) -> None:
        if not choices:
            raise ValueError("Options requires at least one choice")
        self.choices = choices

    def __repr__(self) -> str:
        return f"Options{self.choices!r}"


class Desc(ParamMeta):
    """Human-readable parameter description.

    Parameters
    ----------
    text : str
        Description text shown in GUIs and documentation.
    """

    __slots__ = ('text',)

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return f"Desc({self.text!r})"


# =====================================================================
# ParamSpec — processed introspection data class
# =====================================================================

_SENTINEL = object()


class ParamSpec:
    """Resolved specification for a single tunable parameter.

    Built automatically from ``Annotated`` declarations by
    ``collect_param_specs``.  Attribute names intentionally match the old
    ``TunableParameterSpec`` so that ``grdk.widgets._param_controls``
    works with minimal changes.

    Attributes
    ----------
    name : str
        Parameter name (keyword-argument key).
    param_type : type
        Expected Python type (``float``, ``int``, ``str``, ``bool``, …).
    default : Any
        Default value, or *_SENTINEL* if the parameter is required.
    description : str
        Human-readable description.
    min_value : int, float, or None
        Inclusive minimum (from ``Range``).
    max_value : int, float, or None
        Inclusive maximum (from ``Range``).
    choices : tuple or None
        Allowed values (from ``Options``).
    """

    __slots__ = (
        'name', 'param_type', 'default', '_has_default',
        'description', 'min_value', 'max_value', 'choices',
    )

    def __init__(
        self,
        name: str,
        param_type: type,
        default: Any,
        has_default: bool,
        description: str,
        min_value: Optional[Union[int, float]],
        max_value: Optional[Union[int, float]],
        choices: Optional[Tuple],
    ) -> None:
        self.name = name
        self.param_type = param_type
        self.default = default
        self._has_default = has_default
        self.description = description
        self.min_value = min_value
        self.max_value = max_value
        self.choices = choices

    @property
    def required(self) -> bool:
        """Whether this parameter is required (has no default)."""
        return not self._has_default

    def validate(self, value: Any) -> None:
        """Validate *value* against this spec's type and constraints.

        Rules match the old ``ImageProcessor._validate_tunable_parameters``:

        * ``int`` is accepted when ``param_type`` is ``float``.
        * Range bounds are inclusive.
        * Type-check is skipped when ``param_type`` is ``object``.

        Raises
        ------
        TypeError
            If *value* has the wrong type.
        ValueError
            If *value* violates range or choices constraints.
        """
        # -- type check --
        if self.param_type is float:
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Parameter '{self.name}' must be "
                    f"{self.param_type.__name__}, got {type(value).__name__}"
                )
        elif self.param_type is not object:
            if not isinstance(value, self.param_type):
                raise TypeError(
                    f"Parameter '{self.name}' must be "
                    f"{self.param_type.__name__}, got {type(value).__name__}"
                )

        # -- range check --
        if self.min_value is not None and value < self.min_value:
            raise ValueError(
                f"Parameter '{self.name}' value {value!r} "
                f"is below minimum {self.min_value!r}"
            )
        if self.max_value is not None and value > self.max_value:
            raise ValueError(
                f"Parameter '{self.name}' value {value!r} "
                f"is above maximum {self.max_value!r}"
            )

        # -- choices check --
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"Parameter '{self.name}' value {value!r} "
                f"is not in allowed choices {self.choices!r}"
            )

    def __repr__(self) -> str:
        parts = (
            f"ParamSpec(name={self.name!r}, "
            f"param_type={self.param_type.__name__}, "
            f"required={self.required!r}"
        )
        if not self.required:
            parts += f", default={self.default!r}"
        if self.min_value is not None:
            parts += f", min_value={self.min_value!r}"
        if self.max_value is not None:
            parts += f", max_value={self.max_value!r}"
        if self.choices is not None:
            parts += f", choices={self.choices!r}"
        return parts + ")"


# =====================================================================
# Annotation collection
# =====================================================================

def collect_param_specs(cls: type) -> Tuple[ParamSpec, ...]:
    """Parse ``Annotated`` type hints on *cls* into a tuple of ``ParamSpec``.

    Only fields whose ``Annotated`` metadata includes at least one
    ``ParamMeta`` subclass instance are collected.  Fields are ordered by
    MRO (parent-first, preserving declaration order within each class).

    Raises
    ------
    TypeError
        If a field has both ``Range`` and ``Options`` constraints
        (mutually exclusive).
    """
    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        return ()

    # Determine field ordering: walk MRO bottom-up, collect names in
    # order of first appearance, then reverse so parents come first.
    seen: set = set()
    ordered_names: list = []
    for klass in reversed(cls.__mro__):
        for name in getattr(klass, '__annotations__', {}):
            if name not in seen and name in hints:
                seen.add(name)
                ordered_names.append(name)

    specs: list = []
    for name in ordered_names:
        hint = hints[name]
        if get_origin(hint) is not Annotated:
            continue

        base_type = hint.__args__[0]
        metadata = hint.__metadata__

        # Filter to ParamMeta instances
        param_metas = [m for m in metadata if isinstance(m, ParamMeta)]
        if not param_metas:
            continue

        # Extract constraint types
        range_meta: Optional[Range] = None
        options_meta: Optional[Options] = None
        desc_meta: Optional[Desc] = None
        for m in param_metas:
            if isinstance(m, Range):
                range_meta = m
            elif isinstance(m, Options):
                options_meta = m
            elif isinstance(m, Desc):
                desc_meta = m

        # Mutual exclusivity
        if range_meta and options_meta:
            raise TypeError(
                f"Parameter '{name}' on {cls.__qualname__}: "
                f"Range and Options are mutually exclusive."
            )

        # Resolve default from class attribute
        default = getattr(cls, name, _SENTINEL)
        has_default = default is not _SENTINEL

        specs.append(ParamSpec(
            name=name,
            param_type=base_type,
            default=default if has_default else None,
            has_default=has_default,
            description=desc_meta.text if desc_meta else '',
            min_value=range_meta.min if range_meta else None,
            max_value=range_meta.max if range_meta else None,
            choices=options_meta.choices if options_meta else None,
        ))

    return tuple(specs)


# =====================================================================
# __init__ generation
# =====================================================================

def _make_init(param_specs: Tuple[ParamSpec, ...]):
    """Build an ``__init__`` from *param_specs* with a proper signature.

    The generated function:

    1. Accepts keyword-only arguments matching each spec.
    2. Falls back to the spec default when a kwarg is absent.
    3. Validates every value via ``spec.validate(value)``.
    4. Sets ``self.<name> = value``.
    5. Calls ``self.__post_init__()`` if the class defines one.
    """
    # Pre-build the list of (spec, has_default) for the closure.
    _specs = param_specs

    def __init__(self, **kwargs):
        for spec in _specs:
            if spec.name in kwargs:
                value = kwargs[spec.name]
            elif spec._has_default:
                value = spec.default
            else:
                raise TypeError(
                    f"{type(self).__name__}() missing required "
                    f"keyword argument: '{spec.name}'"
                )
            spec.validate(value)
            object.__setattr__(self, spec.name, value)

        # Check for unexpected keyword arguments
        expected = {s.name for s in _specs}
        unexpected = set(kwargs) - expected
        if unexpected:
            raise TypeError(
                f"{type(self).__name__}() got unexpected "
                f"keyword arguments: {', '.join(sorted(unexpected))}"
            )

        if hasattr(self, '__post_init__'):
            self.__post_init__()

    # Set a proper inspect.Signature so IDEs show real params.
    params = [inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for spec in _specs:
        if spec._has_default:
            params.append(inspect.Parameter(
                spec.name,
                inspect.Parameter.KEYWORD_ONLY,
                default=spec.default,
            ))
        else:
            params.append(inspect.Parameter(
                spec.name,
                inspect.Parameter.KEYWORD_ONLY,
            ))
    __init__.__signature__ = inspect.Signature(params)
    __init__.__qualname__ = '__init__'

    return __init__
