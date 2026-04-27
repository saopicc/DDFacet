'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

# Took the following from master
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("DDFacet")
    except PackageNotFoundError:
        __version__ = "dev"
except (ImportError, ModuleNotFoundError):
    import pkg_resources
    try:
        __version__ = pkg_resources.require("DDFacet")[0].version
    except pkg_resources.DistributionNotFound:
        __version__ = "dev"

# this was in hackathon-rennes branch
# =======
# # https://github.com/python-poetry/poetry/issues/273#issuecomment-1877789967
# from typing import Any
# import importlib.metadata
# from pathlib import Path

# __package_version = "unknown"


# def __get_package_version() -> str:
#     """Find the version of this package."""
#     global __package_version

#     if __package_version != "unknown":
#         # We already set it at some point in the past,
#         # so return that previous value without any
#         # extra work.
#         return __package_version

#     try:
#         # Try to get the version of the current package if
#         # it is running from a distribution.
#         __package_version = importlib.metadata.version("ddfacet")
#     except importlib.metadata.PackageNotFoundError:
#         # Fall back on getting it from a local pyproject.toml.
#         # This works in a development environment where the
#         # package has not been installed from a distribution.
#         import toml

#         pyproject_toml_file = Path(__file__).parent / "pyproject.toml"
#         if pyproject_toml_file.exists() and pyproject_toml_file.is_file():
#             __package_version = toml.load(pyproject_toml_file)["project"][
#                 "version"
#             ]
#             # Indicate it might be locally modified or unreleased.
#             __package_version = __package_version + "+"

#     return __package_version


# def __getattr__(name: str) -> Any:
#     """Get package attributes."""
#     if name in ("version", "__version__"):
#         return __get_package_version()
#     else:
#         raise AttributeError(f"No attribute {name} in module {__name__}.")

# >>>>>>> HackathonRennes
