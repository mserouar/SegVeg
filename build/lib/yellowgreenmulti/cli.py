"""
Command line interface for the yellowgreen-multi module.
"""
import argparse
import logging
import sys

import yellowgreenmulti
from . import cliutils
from . import yellowgreenmultiutils
import pathlib

_LOGGER = logging.getLogger(__name__)


def create_parser():
	"""
	Create the parser for the yellowgreen-multi command.
	:return: Configured parser.
	:rtype: argparse.ArgumentParser
	"""
	parser = argparse.ArgumentParser(
		description='''
2 class Green/NonGreen pixel level store the results in segmenation folder
'''
		)
	parser.add_argument(
		'-v', '--version',
		action='version',
		version='%(prog)s v{}'.format(yellowgreenmulti.__version__)
		)

	# Positional arguments, declaration order is important
	parser.add_argument(
		'input_folder',
		type=cliutils.sanitize_path,
		help='Directory of the session you want to proces ex: "/mnt/PROCESSDATA/TEMP_TEST_DEV/Simon_test/Session 2021-02-22 09-44-37/" '
		)

	# Positional arguments, declaration order is important
	parser.add_argument(
		'configuration_file',
		type=cliutils.sanitize_path,
		help='Configuration file for the modules ex : "/home/capte-gpu-1/Documents/espaces_personnel/SIMON/modules_Arvalis/yellowgreen-multi/config/yellowConfiguration.json" '
		)

	# Optional arguments
	cliutils.add_boolean_flag(parser, 'debug', 'Enable debug outputs. Imply --verbose.')
	cliutils.add_boolean_flag(parser, 'verbose', 'Enable debug logging.')
	return parser

def main(args=None):
	"""
	Run the main procedure.
	:param list args: List of arguments for the command line interface. If not set, arguments are
		taken from ``sys.argv``.
	"""
	parser = create_parser()
	args = parser.parse_args(args)
	args.verbose = args.verbose or args.debug

	# Ensure the directory exists to create the log file
	log_folder = pathlib.Path(pathlib.PurePath(args.input_folder, 'log'))
	log_folder.mkdir(parents=True, exist_ok=True)
	log_filename = log_folder.joinpath('yellowgreen-multi.log')

	cliutils.setup_logging(debug=args.verbose, filename=log_filename)
	_LOGGER.debug('command: %s', ' '.join(sys.argv))
	_LOGGER.debug('version: %s', yellowgreenmulti.__version__)

	# Call the main function of the module
	yellowgreenmultiutils.yellowclassif(args.input_folder, args.configuration_file)

if __name__ == '__main__':
	main()
