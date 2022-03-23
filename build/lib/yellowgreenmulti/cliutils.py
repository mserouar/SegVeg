"""
Utility classes and functions for the command line interface configuration.
"""
import argparse
import logging
import logging.config
import pathlib
import os
import sys


BOOLEAN_STATES = {
	'0': False, '1': True,
	'false': False, 'true': True,
	'no': False, 'yes': True,
	'off': False, 'on': True
	}


def add_boolean_flag(parser, name, help_str):
	"""
	Helper to add a boolean state flag to the argument parser.

	:param argparse.ArgumentParser parser: Parser to update.
	:param str name: Name of the flag without leading hyphen.
	:param str help_str: Flag help description.
	"""
	parser.add_argument(
		'--{}'.format(name),
		dest=name.replace('-', '_'),
		type=boolean_state,
		default=False,
		const=True,
		nargs='?',
		metavar='BOOL_STATE',
		help=help_str
		)

def boolean_state(value):
	"""
	Convert the given boolean state as a boolean value.

	:param str value: Boolean state to convert.
	:return: Boolean value.
	:rtype: bool
	:raise argparse.ArgumentTypeError: The given value is not an accepted boolean state.
	"""
	try:
		return BOOLEAN_STATES[value.lower()]
	except KeyError:
		raise argparse.ArgumentTypeError('invalid boolean state value: {}'.format(value)) from None

def sanitize_path(path):
	"""
	Sanitize the given path, from the command line interface, and return it normalized and absolute.
	Sanitization includes correctly removing quotes, in case the path contains one or more
	whitespaces.

	:param str path: Path to sanitize.
	:return: Path sanitized, normalized and as absolute.
	:rtype: pathlib.Path
	"""
	return pathlib.Path(os.path.abspath(path.replace("'", '').replace('"', '')))

def setup_logging(debug=False, filename=None):
	"""
	Configure logging for the command line interface.

	:param bool debug: If set, debug logs will also be printed to the terminal.
	:param str filename: Path to the log file to create or extend. If not specified, logging
		to file is deactivated.
	"""
	settings = {
		'version': 1,
		'disable_existing_loggers': False,
		'loggers': {
			'yellowgreenmulti': {
				'level': logging.DEBUG,
				'handlers': ['console', 'console_error']
				}
			},
		'handlers': {
			'console': {
				'class': 'logging.StreamHandler',
				'level': logging.DEBUG if debug else logging.INFO,
				'formatter': 'brief',
				'filters': ['skip_errors'],
				'stream': sys.stdout
				},
			'console_error': {
				'class': 'logging.StreamHandler',
				'level': logging.ERROR,
				'formatter': 'brief',
				'stream': sys.stderr
				}
			},
		'formatters': {
			'brief': {
				'format': '[%(levelname)s] %(message)s'
				},
			'default': {
				'format': '%(asctime)s:%(name)s:%(levelname)s %(message)s',
				'datefmt' : '%Y/%m/%d %H:%M:%S'
				}
			},
		'filters': {
			'skip_errors': {
				'()': LevelFilter,
				'level': logging.ERROR
				}
			}
		}

	# Add file logging if a filename is provided
	if filename is not None:
		settings['root'] = {
			'level': logging.DEBUG,
			'handlers': ['file']
			}
		settings['handlers']['file'] = {
			'class': 'logging.FileHandler',
			'level': logging.DEBUG,
			'formatter': 'default',
			'filename': filename
			}

	logging.config.dictConfig(settings)
	logging.captureWarnings(True)

	# Add a hook to log uncaught exceptions
	def _log_exception_hook(exc_type, exc_value, exc_tb):
		logging.getLogger(__package__).critical(
			'uncaught exception', exc_info=(exc_type, exc_value, exc_tb)
			)
	sys.excepthook = _log_exception_hook


class LevelFilter:
	"""Logging filter used to skip records with the given log level and above."""

	def __init__(self, level):
		"""
		Initialize the filter with a minimum log level to skip.

		:param int level: Logging level to skip, levels that are above are also skipped.
		"""
		self.level = level

	def filter(self, record):
		"""
		Skip the record if its level is equal or above the configured level.

		:param record: Logging record.
		:type record: :class:`logging.LogRecord`
		:return: If this record should be logged.
		:rtype: bool
		"""
		return record.levelno < self.level
