# -*- coding: utf-8 -*-
#  ---
# Copyright (c) 2013, 2016 François Orieux <orieux@l2s.centralesupelec.fr>

# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Commentary:

"""
References
----------

This module provides a mutable numeric float type.

More precisely, the class Real is a container that keep and update a
reference to a float.

The trick is that Real is a numbers.Real subcass and can therefore be
used like a builtin float type (+, - //, round(), ...).

>>> R = Real(1.0)
>>> R + 1
2.0

However a degree of mutability is included in the behaviour, that is
that numeric operator can update the numeric without changing the
object identity. It can then be used as a semantic numeric value that
can change in an algorithm but we can keep a reference to it and it's
meaning, like the 'weight of an object'.

>>> frequency = Real(0.8)
>>> frequency
0.8
>>> frequency2 = frequency + 1.3
>>> frequency2 is frequency
True
>>> frequency
2.1

Presented like this, this is not extremely usefull but the behaviour
can be extended by subclassing, like keeping a trace of all taken
values. This possibility is absolutely not possible even by
subclassing, since builtin float is not mutable. Any operation on a
float lead to a new float with losing the reference.

.. note:: Attention
   When the object is the left operand, the value is updated and self
   returned (mutability). When the object is the right operand, a
   builtin float conversion is done and the mutability (or the
   tracing) is lost. Take care.

.. note:: note
   This class if compatible python 2.7 and python 3.2.3

.. note:: note
   The underlying float value is stored in the .value attribut. The
   attribut, implemented as property, is read/write with type real
   float conversion on the fly.
"""

# code:

from __future__ import (print_function, division,
                        absolute_import, unicode_literals)
import numbers
import math

__author__ = "François Orieux"
__copyright__ = "Copyright (C) 2013, 2015 F. Orieux " \
                "<orieux@l2s.centralesupelec.fr>"
__credits__ = ["François Orieux"]
__license__ = "mit"
__version__ = "1.0.0"
__maintainer__ = "François Orieux"
__email__ = "orieux@l2s.centralesupelec.fr"
__status__ = "stable"
__url__ = "research.orieux.fr"
__keywords__ = "mutable, float, numbers.Real"


class Real(numbers.Real):
    """Real emulate real float builtin number with mutability

    Implements numbers.Real abstract interace
    >>> R = Real(1.0)
    >>> R
    1.0
    >>> isinstance(R, numbers.Real)
    True
    >>> R.real is R
    True
    >>> R.imag is int(0)
    True
    >>> R.conjugate() is R
    True
    >>> complex(R)
    (1+0j)

    Underlying float builtin value is accessible
    >>> type(R.value) is type(2.0)
    True

    Since Real is mutable, left operator update the value
    >>> S = R + 2
    >>> S
    3.0
    >>> S is R
    True
    >>> R += 1
    >>> R
    4.0
    >>> type(R)
    <class '__main__.Real'>

    However, right operator imply float conversion
    >>> S = 2 + R
    >>> S
    6.0
    >>> S is R
    False
    >>> type(S) is type(2.0)
    True

    Of course, addition is possible only with numbers (at least
    Complex from numbers module)
    >>> R + "1"
    Traceback (most recent call last):
       ...
    TypeError: ...
    """
    def __init__(self, value=0.0):
        """Real parameter initialisation

        Real can be instantiated with any numbers.Complex
        subclass and, a fortiori, Real
        >>> R = Real(1)
        >>> R
        1.0
        >>> R2 = Real(R)
        >>> R2
        1.0
        >>> R2 is R
        False

        0 is the default value
        >>> print(Real())
        0.0

        Raise an erro with wrong type manipulation
        >>> Real("a")
        Traceback (most recent call last):
            ...
        AssertionError: Argument must be under numbers.Complex hierarchy
        """
        super(Real, self).__init__()
        assert isinstance(value, numbers.Complex), \
            "Argument must be under numbers.Complex hierarchy"
        self._value = float(value.real)

    @property
    def value(self):
        """.value getter"""
        return self._value

    @value.setter
    def value(self, value: float):
        """.value setter property

        Casting is done
        >>> R = Real(2.0)
        >>> R.value
        2.0
        >>> R.value = 3
        >>> R.value
        3.0
        >>> R.value = "a"
        Traceback (most recent call last):
            ...
        AssertionError: Argument must be under numbers.Complex hierarchy
        """
        assert isinstance(value, numbers.Complex), \
            "Real can't contains object not under numbers.Complex " \
            "hierarchy"
        self._value = float(value.real)

    # Real conversion
    def __float__(self):
        """Float conversion

        >>> f = float(Real(2.0))
        >>> f
        2.0
        >>> type(f) is type(2.0)
        True
        """
        return float(self._value)

    def __str__(self):
        return self._value.__str__()

    def __repr__(self):
        return self.__str__()

    # Arithmetic operators
    def __add__(self, value):
        """The + operator

        Left operator update the value
        >>> R = Real(2)
        >>> S = R + 2
        >>> S
        4.0
        >>> S is R
        True
        >>> R += 1
        >>> R
        5.0
        >>> type(R)
        <class '__main__.Real'>

        >>> R + "1"
        Traceback (most recent call last):
        ...
        TypeError: ...
        """
        if isinstance(value, numbers.Complex):
            self._value = self._value.__add__(float(value.real))
            return self
        else:
            return NotImplemented

    def __radd__(self, value):
        """Right + operator imply float conversion

        >>> R = Real(2)
        >>> S = 2 + R
        >>> S
        4.0
        >>> S is R
        False
        >>> type(S) is type(2.0)
        True

        >>> "a" + R
        Traceback (most recent call last):
        ...
        TypeError: ...
        """
        return self._value.__add__(value)

    def __mul__(self, value):
        """The * operator

        Left operator update the value
        >>> R = Real(2)
        >>> S = R * 2
        >>> S
        4.0
        >>> S is R
        True
        >>> R *= 3
        >>> R
        12.0
        >>> type(R)
        <class '__main__.Real'>

        >>> R * "a"
        Traceback (most recent call last):
        ...
        TypeError: ...
        """
        if isinstance(value, numbers.Complex):
            self._value = self._value.__mul__(float(value.real))
            return self
        else:
            return NotImplemented

    def __rmul__(self, value):
        """Right * operator imply float conversion

        >>> R = Real(2.0)
        >>> S = 2 * R
        >>> S
        4.0
        >>> S is R
        False
        >>> type(S) is type(2.0)
        True

        >>> "a" * R
        Traceback (most recent call last):
        ...
        TypeError: ...
        """
        return self._value.__mul__(value)

    def __div__(self, value):
        """The / operator

        Left operator update the value
        >>> R = Real(1)
        >>> S = R / 2
        >>> S
        0.5
        >>> S is R
        True
        >>> R /= 2
        >>> R
        0.25
        >>> type(R)
        <class '__main__.Real'>

        >>> R / "a"
        Traceback (most recent call last):
        ...
        TypeError: ...
        """
        if isinstance(value, numbers.Complex):
            self._value = self._value.__div__(float(value.real))
            return self
        else:
            return NotImplemented

    def __rdiv__(self, value):
        """Right / operator imply float conversion

        >>> R = Real(2.0)
        >>> S = 1.0 / R
        >>> S
        0.5
        >>> S is R
        False
        >>> type(S) is type(2.0)
        True

        >>> "a" / R
        Traceback (most recent call last):
        ...
        TypeError: ...

        __rtruediv__ is implemented so
        >>> 1 / R
        0.5
        """
        if isinstance(value, numbers.Complex):
            return value.__div__(self._value)
        else:
            return NotImplemented

    def __floordiv__(self, value):
        """The // operator

        Left operator update the value
        >>> R = Real(3)
        >>> S = R // 2
        >>> S
        1.0
        >>> S is R
        True
        >>> R //= 2
        >>> R
        0.0
        >>> type(R)
        <class '__main__.Real'>

        >>> R // "a"
        Traceback (most recent call last):
        ...
        TypeError: ...
        """
        if isinstance(value, numbers.Real):
            self._value = self._value.__floordiv__(float(value))
            return self
        else:
            return NotImplemented

    def __rfloordiv__(self, value):
        """Right // operator imply float conversion

        >>> R = Real(2.0)
        >>> S = 5 // R
        >>> S
        2.0
        >>> S is R
        False
        >>> type(S) is type(2.0)
        True

        >>> "a" // R
        Traceback (most recent call last):
        ...
        TypeError: ...
        """
        if isinstance(value, numbers.Real):
            return float(value).__floordiv__(self._value)
        else:
            return NotImplemented

    def __mod__(self, value):
        """The % operator

        Left operator update the value
        >>> R = Real(9)
        >>> S = R % 7
        >>> S
        2.0
        >>> S is R
        True
        >>> R %= 4
        >>> R
        2.0
        >>> type(R)
        <class '__main__.Real'>

        >>> R % "a"
        Traceback (most recent call last):
        ...
        TypeError: ...
        """
        if isinstance(value, numbers.Real):
            self._value = self._value.__mod__(float(value))
            return self
        else:
            return NotImplemented

    def __rmod__(self, value):
        """Right % operator imply float conversion
        >>> R = Real(6.0)
        >>> S = 7 % R
        >>> S
        1.0
        >>> S is R
        False
        >>> type(S) is type(2.0)
        True

        >>> ("a" % R) == 'a'
        True
        """
        if isinstance(value, numbers.Real):
            return float(value).__mod__(self._value)
        else:
            return NotImplemented

    def __pow__(self, value):
        """The ** operator

        Left operator update the value
        >>> R = Real(3)
        >>> S = R ** 2
        >>> S
        9.0
        >>> S is R
        True
        >>> R **= 2
        >>> R
        81.0
        >>> type(R)
        <class '__main__.Real'>

        >>> R ** "a"
        Traceback (most recent call last):
        ...
        TypeError: ...
        """
        if isinstance(value, numbers.Real):
            self._value = self._value.__pow__(float(value))
            return self
        else:
            return NotImplemented

    def __rpow__(self, value):
        """Right ** operator imply float conversion
        >>> R = Real(3.0)
        >>> S = 2 ** R
        >>> S
        8.0
        >>> S is R
        False
        >>> type(S) is type(2.0)
        True

        >>> "a" ** R
        Traceback (most recent call last):
        ...
        TypeError: ...
        """
        if isinstance(value, numbers.Real):
            return float(value.real).__pow__(self._value)
        else:
            return NotImplemented

    def __truediv__(self, value):
        """The / operator

        Left operator update the value
        >>> R = Real(5)
        >>> S = R / 2
        >>> S
        2.5
        >>> S is R
        True
        >>> R /= 2
        >>> R
        1.25
        >>> type(R)
        <class '__main__.Real'>

        >>> R / "a"
        Traceback (most recent call last):
        ...
        TypeError: ...
        """
        if isinstance(value, numbers.Complex):
            self._value = self._value.__truediv__(float(value.real))
            return self
        else:
            return NotImplemented

    def __rtruediv__(self, value):
        """Right / operator imply float conversion

        >>> R = Real(2.0)
        >>> S = 1 / R
        >>> S
        0.5
        >>> S is R
        False
        >>> type(S) is type(2.0)
        True

        >>> "a" / R
        Traceback (most recent call last):
        ...
        TypeError: ...
        """
        if isinstance(value, numbers.Complex):
            return float(value.real).__truediv__(self._value)
        else:
            return NotImplemented

    # Unary operator
    def __neg__(self):
        """Unary negation - operator

        >>> R = Real(2.0)
        >>> -R
        -2.0
        >>> S = (-R)
        >>> S is R
        True
        """
        self._value = self._value.__neg__()
        return self

    def __pos__(self):
        """Unary + operator

        >>> R = Real(-2.0)
        >>> R
        -2.0
        >>> +R
        -2.0
        """
        return self

    def __abs__(self):
        """Absolute value

        >>> R = Real(-2.0)
        >>> abs(R)
        2.0
        >>> R is abs(R)
        True
        """
        self._value = self._value.__abs__()
        return self

    def __trunc__(self):
        """Truncation (math.trunc(), int())

        >>> R = Real(2.0)
        >>> int(R)
        2
        >>> "a" * int(R) == 'aa'
        True
        """
        return self._value.__trunc__()

    def __floor__(self):
        """Reply to math.floor()

        >>> import math
        >>> R = Real(2.7)
        >>> math.floor(R) == 2
        True
        """
        return math.floor(self._value)

    def __ceil__(self):
        """Reply to math.ceil()

        >>> import math
        >>> R = Real(2.2)
        >>> math.ceil(R) == 3
        True
        """
        return math.ceil(self._value)

    def __round__(self):
        """Reply to round()

        >>> round(Real(2.2)) == 2
        True
        >>> round(Real(2.5)) == round(2.5)
        True
        >>> round(Real(2.7)) == 3
        True
        """
        return round(self._value)

    # Comparison operator
    def __eq__(self, value):
        """Equality ==

        >>> R = Real(2.0)
        >>> R == 2.0
        True
        >>> R != 1.0
        True
        >>> 3.0 == R
        False
        >>> "a" == R
        False
        >>> R == [1]
        False
        >>> R == Real(2.0)
        True
        """
        if isinstance(value, Real):
            return self._value.__eq__(value.value)
        else:
            return self._value.__eq__(value)

    def __le__(self, value):
        """Inferior or equal <=

        >>> R = Real(2.0)
        >>> R <= 2.0
        True
        >>> R <= 3.0
        True
        >>> R <= Real(2.0)
        True
        >>> R <= Real(1.0)
        False
        """
        if isinstance(value, Real):
            return self._value.__le__(value.value)
        return self._value.__le__(value)

    def __ge__(self, value):
        """Superior or equal >=

        >>> R = Real(2.0)
        >>> R >= 0.0
        True
        >>> 3.0 <= R
        False
        >>> R >= Real(1.0)
        True
        >>> R >= Real(3.0)
        False
        """
        if isinstance(value, Real):
            return self._value.__ge__(value.value)
        return self._value.__ge__(value)

    def __lt__(self, value):
        """Inferior <

        >>> R = Real(2.0)
        >>> R < 2.0
        False
        >>> 3 < R
        False
        >>> R < 3.0
        True
        >>> R < Real(3.0)
        True
        """
        if isinstance(value, Real):
            return self._value.__lt__(value.value)
        return self._value.__lt__(value)

    def __gt__(self, value):
        """Superior >

        >>> R = Real(2.0)
        >>> R > 0.0
        True
        >>> R > 3.0
        False
        >>> R > Real(1.0)
        True
        """
        if isinstance(value, Real):
            return self._value.__gt__(value.value)
        return self._value.__gt__(value)

    def __ilshift__(self, value):
        """Assignement revisited <<=

        Update the value of the variable self
        >>> R = Real(2.0)
        >>> R <<= 5
        >>> R == 5.0
        True

        Notes
        -----
        I'm not sure it's a good idea to change the meaning of this
        operator (left bitwise shift). Maybe it will change in the future.
        """
        self.value = value
        return self


class RealHist(Real):
    def __init__(self, value=0.0):
        super(RealHist, self).__init__(value)
        self.trace = [self._value]

    @property
    def value(self):
        """.value getter"""
        return self._value

    @value.setter
    def value(self, value):
        """.value setter property

        Casting is done
        >>> R = Real(2.0)
        >>> R.value
        2.0
        >>> R.value = 3
        >>> R.value
        3.0
        >>> R.value = "a"
        Traceback (most recent call last):
            ...
        AssertionError: Argument must be under numbers.Complex hierarchy
        """
        assert isinstance(value, numbers.Complex), \
            "Real can't contains object not under numbers.Complex " \
            "hierarchy"
        self._value = float(value.real)
        self.trace.append(self._value)


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
