#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# =============================================================================
#  Version: 1.00 (December 15, 2015)
#  Author: Giuseppe Attardi (attardi@di.unipi.it), University of Pisa
#
# =============================================================================
#  Copyright (c) 2015. Giuseppe Attardi (attardi@di.unipi.it).
# =============================================================================
#  This file is part of Tanl.
#
#  Tanl is free software; you can redistribute it and/or modify it
#  under the terms of the GNU General Public License, version 3,
#  as published by the Free Software Foundation.
#
#  Tanl is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

"""Wikipedia Cirrus Extractor:
Extracts and cleans text from a Wikipedia Cirrus dump and stores output in a
number of files of similar size in a given directory.
Each file will contain several documents in the format:

	<doc id="" url="" title="" language="" revision="">
        ...
        </doc>

"""

import sys, os.path, time
import re
import json
import argparse
import bz2
import gzip
import logging

# Program version
version = '1.00'

urlbase = 'http://it.wikipedia.org/'

# ----------------------------------------------------------------------

class NextFile(object):
    """
    Synchronous generation of next available file name.
    """

    filesPerDir = 100

    def __init__(self, path_name):
        self.path_name = path_name
        self.dir_index = -1
        self.file_index = -1

    def next(self):
        self.file_index = (self.file_index + 1) % NextFile.filesPerDir
        if self.file_index == 0:
            self.dir_index += 1
        dirname = self._dirname()
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        return self._filepath()

    def _dirname(self):
        char1 = self.dir_index % 26
        char2 = self.dir_index / 26 % 26
        return os.path.join(self.path_name, '%c%c' % (ord('A') + char2, ord('A') + char1))

    def _filepath(self):
        return '%s/wiki_%02d' % (self._dirname(), self.file_index)

class OutputSplitter(object):
    """
    File-like object, that splits output to multiple files of a given max size.
    """

    def __init__(self, nextFile, max_file_size=0, compress=True):
        """
        :param nextfile: a NextFile object from which to obtain filenames
            to use.
        :param max_file_size: the maximum size of each file.
        :para compress: whether to write data with bzip compression.
        """
        self.nextFile = nextFile
        self.compress = compress
        self.max_file_size = max_file_size
        self.file = self.open(self.nextFile.next())

    def reserve(self, size):
        if self.file.tell() + size > self.max_file_size:
            self.close()
            self.file = self.open(self.nextFile.next())

    def write(self, data):
        self.reserve(len(data))
        self.file.write(data)

    def close(self):
        self.file.close()

    def open(self, filename):
        if self.compress:
            return bz2.BZ2File(filename + '.bz2', 'w')
        else:
            return open(filename, 'w')

# ----------------------------------------------------------------------

class Extractor(object):

    def extract(self, out):
        """
        :param out: output file.
        """
        logging.debug("%s\t%s", self.id, self.title)
        text = ''.join(self.page)
        url = get_url(self.id)
        header = '<doc id="%s" url="%s" title="%s" language="%s" revision="%s">\n' % (self.id, url, self.title, self.language, self.revision)
        # Separate header from text with a newline.
        header += self.title + '\n\n'
        header = header.encode('utf-8')
        footer = "\n</doc>\n"
        out.write(header)
        text = clean(self, text)
        for line in compact(text):
            out.write(line.encode('utf-8'))
            out.write('\n')
        out.write(footer)

def process_dump(input_file, out_file, file_size, file_compress):
    """
    :param input_file: name of the wikipedia dump file; '-' to read from stdin
    :param out_file: directory where to store extracted data, or '-' for stdout
    :param file_size: max size of each extracted file, or None for no max (one file)
    :param file_compress: whether to compress files with bzip.
    """

    if input_file == '-':
        input = sys.stdin
    else:
        input = gzip.open(input_file)

    if out_file == '-':
        output = sys.stdout
        if file_compress:
            logging.warn("writing to stdout, so no output compression (use external tool)")
    else:
        nextFile = NextFile(out_file)
        output = OutputSplitter(nextFile, file_size, file_compress)

    # process dump
    # format
    # {"index":{"_type":"page","_id":"3825914"}}
    # {"namespace":0,"title":TITLE,"timestamp":"2014-06-29T15:51:09Z","text":TEXT,...}
    while True:
        line = input.readline()
        if not line:
            break
        index = json.loads(line)
        content = json.loads(input.readline())
        type = index['index']['_type']
        id = index['index']['_id']
        language = content['language']
        revision = content['version']
        if type == 'page' and content['namespace'] == 0:
            title = content['title']
            text = content['text']
            # drop references:
            # ^ The Penguin Dictionary
            text = re.sub(r'  \^ .*', '', text)
            url = urlbase + 'wiki?curid=' + id
            header = '<doc id="%s" url="%s" title="%s" language="%s" revision="%s">\n' % (id, url, title, language, revision)
            page = header + title + '\n\n' + text + '\n</doc>\n'
            output.write(page.encode('utf-8'))

# ----------------------------------------------------------------------

# Minimum size of output files
minFileSize = 200 * 1024

def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument("input",
                        help="Cirrus Json wiki dump file")
    groupO = parser.add_argument_group('Output')
    groupO.add_argument("-o", "--output", default="text",
                        help="directory for extracted files (or '-' for dumping to stdin)")
    groupO.add_argument("-b", "--bytes", default="1M",
                        help="maximum bytes per output file (default %(default)s)",
                        metavar="n[KMG]")
    groupO.add_argument("-c", "--compress", action="store_true",
                        help="compress output files using bzip")

    groupP = parser.add_argument_group('Processing')
    groupP.add_argument("-ns", "--namespaces", default="", metavar="ns1,ns2",
                        help="accepted namespaces")

    groupS = parser.add_argument_group('Special')
    groupS.add_argument("-q", "--quiet", action="store_true",
                        help="suppress reporting progress info")
    groupS.add_argument("-v", "--version", action="version",
                        version='%(prog)s ' + version,
                        help="print program version")

    args = parser.parse_args()

    try:
        power = 'kmg'.find(args.bytes[-1].lower()) + 1
        file_size = int(args.bytes[:-1]) * 1024 ** power
        if file_size < minFileSize:
            raise ValueError()
    except ValueError:
        logging.error('Insufficient or invalid size: %s', args.bytes)
        return

    FORMAT = '%(levelname)s: %(message)s'
    logging.basicConfig(format=FORMAT)

    logger = logging.getLogger()
    if not args.quiet:
        logger.setLevel(logging.INFO)

    input_file = args.input

    output_path = args.output
    if output_path != '-' and not os.path.isdir(output_path):
        try:
            os.makedirs(output_path)
        except:
            logging.error('Could not create: %s', output_path)
            return

    process_dump(input_file, output_path, file_size, args.compress)


if __name__ == '__main__':
    main()
