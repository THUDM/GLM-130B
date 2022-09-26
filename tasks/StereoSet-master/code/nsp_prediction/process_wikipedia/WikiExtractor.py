#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
#  Version: 2.75 (March 4, 2017)
#  Author: Giuseppe Attardi (attardi@di.unipi.it), University of Pisa
#
#  Contributors:
#   Antonio Fuschetto (fuschett@aol.com)
#   Leonardo Souza (lsouza@amtera.com.br)
#   Juan Manuel Caicedo (juan@cavorite.com)
#   Humberto Pereira (begini@gmail.com)
#   Siegfried-A. Gevatter (siegfried@gevatter.com)
#   Pedro Assis (pedroh2306@gmail.com)
#   Wim Muskee (wimmuskee@gmail.com)
#   Radics Geza (radicsge@gmail.com)
#   orangain (orangain@gmail.com)
#   Seth Cleveland (scleveland@turnitin.com)
#   Bren Barn
#
# =============================================================================
#  Copyright (c) 2011-2017. Giuseppe Attardi (attardi@di.unipi.it).
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
#  GNU General Public License at <http://www.gnu.org/licenses/> for more details.
#
# =============================================================================

"""Wikipedia Extractor:
Extracts and cleans text from a Wikipedia database dump and stores output in a
number of files of similar size in a given directory.
Each file will contain several documents in the format:

    <doc id="" revid="" url="" title="">
        ...
        </doc>

If the program is invoked with the --json flag, then each file will
contain several documents formatted as json ojects, one per line, with
the following structure

    {"id": "", "revid": "", "url":"", "title": "", "text": "..."}

Template expansion requires preprocesssng first the whole dump and
collecting template definitions.

"""

from __future__ import unicode_literals, division

import sys
import argparse
import bz2
import codecs
import cgi
import fileinput
import logging
import os.path
import re  # TODO use regex when it will be standard
import time
import json
from io import StringIO
from multiprocessing import Queue, Process, Value, cpu_count
from timeit import default_timer


PY2 = sys.version_info[0] == 2
# Python 2.7 compatibiity
if PY2:
    from urllib import quote
    from htmlentitydefs import name2codepoint
    from itertools import izip as zip, izip_longest as zip_longest
    range = xrange  # Use Python 3 equivalent
    chr = unichr    # Use Python 3 equivalent
    text_type = unicode

    class SimpleNamespace(object):
        def __init__ (self, **kwargs):
            self.__dict__.update(kwargs)
        def __repr__ (self):
            keys = sorted(self.__dict__)
            items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
            return "{}({})".format(type(self).__name__, ", ".join(items))
        def __eq__ (self, other):
            return self.__dict__ == other.__dict__
else:
    from urllib.parse import quote
    from html.entities import name2codepoint
    from itertools import zip_longest
    from types import SimpleNamespace
    text_type = str


# ===========================================================================

# Program version
version = '2.75'

## PARAMS ####################################################################

options = SimpleNamespace(

    ##
    # Defined in <siteinfo>
    # We include as default Template, when loading external template file.
    knownNamespaces = {'Template': 10},

    ##
    # The namespace used for template definitions
    # It is the name associated with namespace key=10 in the siteinfo header.
    templateNamespace = '',
    templatePrefix = '',

    ##
    # The namespace used for module definitions
    # It is the name associated with namespace key=828 in the siteinfo header.
    moduleNamespace = '',

    ##
    # Recognize only these namespaces in links
    # w: Internal links to the Wikipedia
    # wiktionary: Wiki dictionary
    # wikt: shortcut for Wiktionary
    #
    acceptedNamespaces = ['w', 'wiktionary', 'wikt'],

    # This is obtained from <siteinfo>
    urlbase = '',

    ##
    # Filter disambiguation pages
    filter_disambig_pages = False,

    ##
    # Drop tables from the article
    keep_tables = False,

    ##
    # Whether to preserve links in output
    keepLinks = False,

    ##
    # Whether to preserve section titles
    keepSections = True,

    ##
    # Whether to preserve lists
    keepLists = False,

    ##
    # Whether to output HTML instead of text
    toHTML = False,

    ##
    # Whether to write json instead of the xml-like default output format
    write_json = False,

    ##
    # Whether to expand templates
    expand_templates = True,

    ##
    ## Whether to escape doc content
    escape_doc = False,

    ##
    # Print the wikipedia article revision
    print_revision = False,

    ##
    # Minimum expanded text length required to print document
    min_text_length = 0,

    # Shared objects holding templates, redirects and cache
    templates = {},
    redirects = {},
    # cache of parser templates
    # FIXME: sharing this with a Manager slows down.
    templateCache = {},

    # Elements to ignore/discard

    ignored_tag_patterns = [],
    filter_category_include = set(),
    filter_category_exclude = set(),

    log_file = None,

    discardElements = [
        'gallery', 'timeline', 'noinclude', 'pre',
        'table', 'tr', 'td', 'th', 'caption', 'div',
        'form', 'input', 'select', 'option', 'textarea',
        'ul', 'li', 'ol', 'dl', 'dt', 'dd', 'menu', 'dir',
        'ref', 'references', 'img', 'imagemap', 'source', 'small',
        'sub', 'sup', 'indicator'
    ],
)

##
# Keys for Template and Module namespaces
templateKeys = set(['10', '828'])

##
# Regex for identifying disambig pages
filter_disambig_page_pattern = re.compile("{{disambig(uation)?(\|[^}]*)?}}|__DISAMBIG__")

##
g_page_total = 0
g_page_articl_total=0
g_page_articl_used_total=0
# page filtering logic -- remove templates, undesired xml namespaces, and disambiguation pages
def keepPage(ns, catSet, page):
    global g_page_articl_total,g_page_total,g_page_articl_used_total
    g_page_total += 1
    if ns != '0':               # Aritcle
        return False
    # remove disambig pages if desired
    g_page_articl_total += 1
    if options.filter_disambig_pages:
        for line in page:
            if filter_disambig_page_pattern.match(line):
                return False
    if len(options.filter_category_include) > 0 and len(options.filter_category_include & catSet)==0:
        logging.debug("***No include  " + str(catSet))
        return False
    if len(options.filter_category_exclude) > 0 and len(options.filter_category_exclude & catSet)>0:
        logging.debug("***Exclude  " + str(catSet))
        return False
    g_page_articl_used_total += 1
    return True


def get_url(uid):
    return "%s?curid=%s" % (options.urlbase, uid)


# =========================================================================
#
# MediaWiki Markup Grammar
# https://www.mediawiki.org/wiki/Preprocessor_ABNF

# xml-char = %x9 / %xA / %xD / %x20-D7FF / %xE000-FFFD / %x10000-10FFFF
# sptab = SP / HTAB

# ; everything except ">" (%x3E)
# attr-char = %x9 / %xA / %xD / %x20-3D / %x3F-D7FF / %xE000-FFFD / %x10000-10FFFF

# literal         = *xml-char
# title           = wikitext-L3
# part-name       = wikitext-L3
# part-value      = wikitext-L3
# part            = ( part-name "=" part-value ) / ( part-value )
# parts           = [ title *( "|" part ) ]
# tplarg          = "{{{" parts "}}}"
# template        = "{{" parts "}}"
# link            = "[[" wikitext-L3 "]]"

# comment         = "<!--" literal "-->"
# unclosed-comment = "<!--" literal END
# ; the + in the line-eating-comment rule was absent between MW 1.12 and MW 1.22
# line-eating-comment = LF LINE-START *SP +( comment *SP ) LINE-END

# attr            = *attr-char
# nowiki-element  = "<nowiki" attr ( "/>" / ( ">" literal ( "</nowiki>" / END ) ) )

# wikitext-L2     = heading / wikitext-L3 / *wikitext-L2
# wikitext-L3     = literal / template / tplarg / link / comment /
#                   line-eating-comment / unclosed-comment / xmlish-element /
#                   *wikitext-L3

# ------------------------------------------------------------------------------

selfClosingTags = ('br', 'hr', 'nobr', 'ref', 'references', 'nowiki')

placeholder_tags = {'math': 'formula', 'code': 'codice'}


def normalizeTitle(title):
    """Normalize title"""
    # remove leading/trailing whitespace and underscores
    title = title.strip(' _')
    # replace sequences of whitespace and underscore chars with a single space
    title = re.sub(r'[\s_]+', ' ', title)

    m = re.match(r'([^:]*):(\s*)(\S(?:.*))', title)
    if m:
        prefix = m.group(1)
        if m.group(2):
            optionalWhitespace = ' '
        else:
            optionalWhitespace = ''
        rest = m.group(3)

        ns = normalizeNamespace(prefix)
        if ns in options.knownNamespaces:
            # If the prefix designates a known namespace, then it might be
            # followed by optional whitespace that should be removed to get
            # the canonical page name
            # (e.g., "Category:  Births" should become "Category:Births").
            title = ns + ":" + ucfirst(rest)
        else:
            # No namespace, just capitalize first letter.
            # If the part before the colon is not a known namespace, then we
            # must not remove the space after the colon (if any), e.g.,
            # "3001: The_Final_Odyssey" != "3001:The_Final_Odyssey".
            # However, to get the canonical page name we must contract multiple
            # spaces into one, because
            # "3001:   The_Final_Odyssey" != "3001: The_Final_Odyssey".
            title = ucfirst(prefix) + ":" + optionalWhitespace + ucfirst(rest)
    else:
        # no namespace, just capitalize first letter
        title = ucfirst(title)
    return title


def unescape(text):
    """
    Removes HTML or XML character references and entities from a text string.

    :param text The HTML (or XML) source text.
    :return The plain text, as a Unicode string, if necessary.
    """

    def fixup(m):
        text = m.group(0)
        code = m.group(1)
        try:
            if text[1] == "#":  # character reference
                if text[2] == "x":
                    return chr(int(code[1:], 16))
                else:
                    return chr(int(code))
            else:  # named entity
                return chr(name2codepoint[code])
        except:
            return text  # leave as is

    return re.sub("&#?(\w+);", fixup, text)


# Match HTML comments
# The buggy template {{Template:T}} has a comment terminating with just "->"
comment = re.compile(r'<!--.*?-->', re.DOTALL)


# Match <nowiki>...</nowiki>
nowiki = re.compile(r'<nowiki>.*?</nowiki>')


def ignoreTag(tag):
    left = re.compile(r'<%s\b.*?>' % tag, re.IGNORECASE | re.DOTALL)  # both <ref> and <reference>
    right = re.compile(r'</\s*%s>' % tag, re.IGNORECASE)
    options.ignored_tag_patterns.append((left, right))

# Match selfClosing HTML tags
selfClosing_tag_patterns = [
    re.compile(r'<\s*%s\b[^>]*/\s*>' % tag, re.DOTALL | re.IGNORECASE) for tag in selfClosingTags
    ]

# Match HTML placeholder tags
placeholder_tag_patterns = [
    (re.compile(r'<\s*%s(\s*| [^>]+?)>.*?<\s*/\s*%s\s*>' % (tag, tag), re.DOTALL | re.IGNORECASE),
     repl) for tag, repl in placeholder_tags.items()
    ]

# Match preformatted lines
preformatted = re.compile(r'^ .*?$')

# Match external links (space separates second optional parameter)
externalLink = re.compile(r'\[\w+[^ ]*? (.*?)]')
externalLinkNoAnchor = re.compile(r'\[\w+[&\]]*\]')

# Matches bold/italic
bold_italic = re.compile(r"'''''(.*?)'''''")
bold = re.compile(r"'''(.*?)'''")
italic_quote = re.compile(r"''\"([^\"]*?)\"''")
italic = re.compile(r"''(.*?)''")
quote_quote = re.compile(r'""([^"]*?)""')

# Matches space
spaces = re.compile(r' {2,}')

# Matches dots
dots = re.compile(r'\.{4,}')


# ======================================================================


class Template(list):
    """
    A Template is a list of TemplateText or TemplateArgs
    """

    @classmethod
    def parse(cls, body):
        tpl = Template()
        # we must handle nesting, s.a.
        # {{{1|{{PAGENAME}}}
        # {{{italics|{{{italic|}}}
        # {{#if:{{{{{#if:{{{nominee|}}}|nominee|candidate}}|}}}|
        #
        start = 0
        for s, e in findMatchingBraces(body, 3):
            tpl.append(TemplateText(body[start:s]))
            tpl.append(TemplateArg(body[s + 3:e - 3]))
            start = e
        tpl.append(TemplateText(body[start:]))  # leftover
        return tpl


    def subst(self, params, extractor, depth=0):
        # We perform parameter substitutions recursively.
        # We also limit the maximum number of iterations to avoid too long or
        # even endless loops (in case of malformed input).

        # :see: http://meta.wikimedia.org/wiki/Help:Expansion#Distinction_between_variables.2C_parser_functions.2C_and_templates
        #
        # Parameter values are assigned to parameters in two (?) passes.
        # Therefore a parameter name in a template can depend on the value of
        # another parameter of the same template, regardless of the order in
        # which they are specified in the template call, for example, using
        # Template:ppp containing "{{{{{{p}}}}}}", {{ppp|p=q|q=r}} and even
        # {{ppp|q=r|p=q}} gives r, but using Template:tvvv containing
        # "{{{{{{{{{p}}}}}}}}}", {{tvvv|p=q|q=r|r=s}} gives s.

        # logging.debug('&*ssubst tpl %d %s', extractor.frame.length, '', depth, self)

        if depth > extractor.maxParameterRecursionLevels:
            extractor.recursion_exceeded_3_errs += 1
            return ''

        return ''.join([tpl.subst(params, extractor, depth) for tpl in self])

    def __str__(self):
        return ''.join([text_type(x) for x in self])


class TemplateText(text_type):
    """Fixed text of template"""


    def subst(self, params, extractor, depth):
        return self


class TemplateArg(object):
    """
    parameter to a template.
    Has a name and a default value, both of which are Templates.
    """

    def __init__(self, parameter):
        """
        :param parameter: the parts of a tplarg.
        """
        # the parameter name itself might contain templates, e.g.:
        #   appointe{{#if:{{{appointer14|}}}|r|d}}14|
        #   4|{{{{{subst|}}}CURRENTYEAR}}

        # any parts in a tplarg after the first (the parameter default) are
        # ignored, and an equals sign in the first part is treated as plain text.
        # logging.debug('TemplateArg %s', parameter)

        parts = splitParts(parameter)
        self.name = Template.parse(parts[0])
        if len(parts) > 1:
            # This parameter has a default value
            self.default = Template.parse(parts[1])
        else:
            self.default = None

    def __str__(self):
        if self.default:
            return '{{{%s|%s}}}' % (self.name, self.default)
        else:
            return '{{{%s}}}' % self.name


    def subst(self, params, extractor, depth):
        """
        Substitute value for this argument from dict :param params:
        Use :param extractor: to evaluate expressions for name and default.
        Limit substitution to the maximun :param depth:.
        """
        # the parameter name itself might contain templates, e.g.:
        # appointe{{#if:{{{appointer14|}}}|r|d}}14|
        paramName = self.name.subst(params, extractor, depth + 1)
        paramName = extractor.transform(paramName)
        res = ''
        if paramName in params:
            res = params[paramName]  # use parameter value specified in template invocation
        elif self.default:  # use the default value
            defaultValue = self.default.subst(params, extractor, depth + 1)
            res = extractor.transform(defaultValue)
        # logging.debug('subst arg %d %s -> %s' % (depth, paramName, res))
        return res


class Frame(object):

    def __init__(self, title='', args=[], prev=None):
        self.title = title
        self.args = args
        self.prev = prev
        self.depth = prev.depth + 1 if prev else 0


    def push(self, title, args):
        return Frame(title, args, self)


    def pop(self):
        return self.prev


    def __str__(self):
        res = ''
        prev = self.prev
        while prev:
            if res: res += ', '
            res += '(%s, %s)' % (prev.title, prev.args)
            prev = prev.prev
        return '<Frame [' + res + ']>'

# ======================================================================

substWords = 'subst:|safesubst:'

class Extractor(object):
    """
    An extraction task on a article.
    """
    def __init__(self, id, revid, title, lines):
        """
        :param id: id of page.
        :param title: tutle of page.
        :param lines: a list of lines.
        """
        self.id = id
        self.revid = revid
        self.title = title
        self.text = ''.join(lines)
        self.magicWords = MagicWords()
        self.frame = Frame()
        self.recursion_exceeded_1_errs = 0  # template recursion within expand()
        self.recursion_exceeded_2_errs = 0  # template recursion within expandTemplate()
        self.recursion_exceeded_3_errs = 0  # parameter recursion
        self.template_title_errs = 0

    def write_output(self, out, text):
        """
        :param out: a memory file
        :param text: the text of the page
        """
        url = get_url(self.id)
        if options.write_json:
            json_data = {
                'id': self.id,
                'url': url,
                'title': self.title,
                'text': "\n".join(text)
            }
            if options.print_revision:
                json_data['revid'] = self.revid
            # We don't use json.dump(data, out) because we want to be
            # able to encode the string if the output is sys.stdout
            out_str = json.dumps(json_data, ensure_ascii=False)
            if out == sys.stdout:   # option -a or -o -
                out_str = out_str.encode('utf-8')
            out.write(out_str)
            out.write('\n')
        else:
            if options.print_revision:
                header = '<doc id="%s" revid="%s" url="%s" title="%s">\n' % (self.id, self.revid, url, self.title)
            else:
                header = '<doc id="%s" url="%s" title="%s">\n' % (self.id, url, self.title)
            footer = "\n</doc>\n"
            if out == sys.stdout:   # option -a or -o -
                header = header.encode('utf-8')
            out.write(header)
            for line in text:
                if out == sys.stdout:   # option -a or -o -
                    line = line.encode('utf-8')
                out.write(line)
                out.write('\n')
            out.write(footer)

    def extract(self, out):
        """
        :param out: a memory file.
        """
        logging.info('%s\t%s', self.id, self.title)

        # Separate header from text with a newline.
        if options.toHTML:
            title_str = '<h1>' + self.title + '</h1>'
        else:
            title_str = self.title + '\n'
        # https://www.mediawiki.org/wiki/Help:Magic_words
        colon = self.title.find(':')
        if colon != -1:
            ns = self.title[:colon]
            pagename = self.title[colon+1:]
        else:
            ns = '' # Main
            pagename = self.title
        self.magicWords['NAMESPACE'] = ns
        self.magicWords['NAMESPACENUMBER'] = options.knownNamespaces.get(ns, '0')
        self.magicWords['PAGENAME'] = pagename
        self.magicWords['FULLPAGENAME'] = self.title
        slash = pagename.rfind('/')
        if slash != -1:
            self.magicWords['BASEPAGENAME'] = pagename[:slash]
            self.magicWords['SUBPAGENAME'] = pagename[slash+1:]
        else:
            self.magicWords['BASEPAGENAME'] = pagename
            self.magicWords['SUBPAGENAME'] = ''
        slash = pagename.find('/')
        if slash != -1:
            self.magicWords['ROOTPAGENAME'] = pagename[:slash]
        else:
            self.magicWords['ROOTPAGENAME'] = pagename
        self.magicWords['CURRENTYEAR'] = time.strftime('%Y')
        self.magicWords['CURRENTMONTH'] = time.strftime('%m')
        self.magicWords['CURRENTDAY'] = time.strftime('%d')
        self.magicWords['CURRENTHOUR'] = time.strftime('%H')
        self.magicWords['CURRENTTIME'] = time.strftime('%H:%M:%S')
        text = self.text
        self.text = ''          # save memory
        #
        # @see https://doc.wikimedia.org/mediawiki-core/master/php/classParser.html
        # This does the equivalent of internalParse():
        #
        # $dom = $this->preprocessToDom( $text, $flag );
        # $text = $frame->expand( $dom );
        #
        text = self.transform(text)
        text = self.wiki2text(text)
        text = compact(self.clean(text))
        # from zwChan
        text = [title_str] + text

        if sum(len(line) for line in text) < options.min_text_length:
            return

        self.write_output(out, text)

        errs = (self.template_title_errs,
                self.recursion_exceeded_1_errs,
                self.recursion_exceeded_2_errs,
                self.recursion_exceeded_3_errs)
        if any(errs):
            logging.warn("Template errors in article '%s' (%s): title(%d) recursion(%d, %d, %d)",
                         self.title, self.id, *errs)


    def transform(self, wikitext):
        """
        Transforms wiki markup.
        @see https://www.mediawiki.org/wiki/Help:Formatting
        """
        # look for matching <nowiki>...</nowiki>
        res = ''
        cur = 0
        for m in nowiki.finditer(wikitext, cur):
            res += self.transform1(wikitext[cur:m.start()]) + wikitext[m.start():m.end()]
            cur = m.end()
        # leftover
        res += self.transform1(wikitext[cur:])
        return res


    def transform1(self, text):
        """Transform text not containing <nowiki>"""
        if options.expand_templates:
            # expand templates
            # See: http://www.mediawiki.org/wiki/Help:Templates
            return self.expand(text)
        else:
            # Drop transclusions (template, parser functions)
            return dropNested(text, r'{{', r'}}')


    def wiki2text(self, text):
        #
        # final part of internalParse().)
        #
        # $text = $this->doTableStuff( $text );
        # $text = preg_replace( '/(^|\n)-----*/', '\\1<hr />', $text );
        # $text = $this->doDoubleUnderscore( $text );
        # $text = $this->doHeadings( $text );
        # $text = $this->replaceInternalLinks( $text );
        # $text = $this->doAllQuotes( $text );
        # $text = $this->replaceExternalLinks( $text );
        # $text = str_replace( self::MARKER_PREFIX . 'NOPARSE', '', $text );
        # $text = $this->doMagicLinks( $text );
        # $text = $this->formatHeadings( $text, $origText, $isMain );

        # Drop tables
        # first drop residual templates, or else empty parameter |} might look like end of table.
        if not options.keep_tables:
            text = dropNested(text, r'{{', r'}}')
            text = dropNested(text, r'{\|', r'\|}')

        # Handle bold/italic/quote
        if options.toHTML:
            text = bold_italic.sub(r'<b>\1</b>', text)
            text = bold.sub(r'<b>\1</b>', text)
            text = italic.sub(r'<i>\1</i>', text)
        else:
            text = bold_italic.sub(r'\1', text)
            text = bold.sub(r'\1', text)
            text = italic_quote.sub(r'"\1"', text)
            text = italic.sub(r'"\1"', text)
            text = quote_quote.sub(r'"\1"', text)
        # residuals of unbalanced quotes
        text = text.replace("'''", '').replace("''", '"')

        # replace internal links
        text = replaceInternalLinks(text)

        # replace external links
        text = replaceExternalLinks(text)

        # drop MagicWords behavioral switches
        text = magicWordsRE.sub('', text)

        # ############### Process HTML ###############

        # turn into HTML, except for the content of <syntaxhighlight>
        res = ''
        cur = 0
        for m in syntaxhighlight.finditer(text):
            res += unescape(text[cur:m.start()]) + m.group(1)
            cur = m.end()
        text = res + unescape(text[cur:])
        return text


    def clean(self, text):
        """
        Removes irrelevant parts from :param: text.
        """

        # Collect spans
        spans = []
        # Drop HTML comments
        for m in comment.finditer(text):
            spans.append((m.start(), m.end()))

        # Drop self-closing tags
        for pattern in selfClosing_tag_patterns:
            for m in pattern.finditer(text):
                spans.append((m.start(), m.end()))

        # Drop ignored tags
        for left, right in options.ignored_tag_patterns:
            for m in left.finditer(text):
                spans.append((m.start(), m.end()))
            for m in right.finditer(text):
                spans.append((m.start(), m.end()))

        # Bulk remove all spans
        text = dropSpans(spans, text)

        # Drop discarded elements
        for tag in options.discardElements:
            text = dropNested(text, r'<\s*%s\b[^>/]*>' % tag, r'<\s*/\s*%s>' % tag)

        if not options.toHTML:
            # Turn into text what is left (&amp;nbsp;) and <syntaxhighlight>
            text = unescape(text)

        # Expand placeholders
        for pattern, placeholder in placeholder_tag_patterns:
            index = 1
            for match in pattern.finditer(text):
                text = text.replace(match.group(), '%s_%d' % (placeholder, index))
                index += 1

        text = text.replace('<<', '«').replace('>>', '»')

        #############################################

        # Cleanup text
        text = text.replace('\t', ' ')
        text = spaces.sub(' ', text)
        text = dots.sub('...', text)
        text = re.sub(' (,:\.\)\]»)', r'\1', text)
        text = re.sub('(\[\(«) ', r'\1', text)
        text = re.sub(r'\n\W+?\n', '\n', text, flags=re.U)  # lines with only punctuations
        text = text.replace(',,', ',').replace(',.', '.')
        if options.keep_tables:
            # the following regular expressions are used to remove the wikiml chartacters around table strucutures
            # yet keep the content. The order here is imporant so we remove certain markup like {| and then
            # then the future html attributes such as 'style'. Finally we drop the remaining '|-' that delimits cells.
            text = re.sub(r'!(?:\s)?style=\"[a-z]+:(?:\d+)%;\"', r'', text)
            text = re.sub(r'!(?:\s)?style="[a-z]+:(?:\d+)%;[a-z]+:(?:#)?(?:[0-9a-z]+)?"', r'', text)
            text = text.replace('|-', '')
            text = text.replace('|', '')
        if options.toHTML:
            text = cgi.escape(text)
        return text


    # ----------------------------------------------------------------------
    # Expand templates

    maxTemplateRecursionLevels = 30
    maxParameterRecursionLevels = 10

    # check for template beginning
    reOpen = re.compile('(?<!{){{(?!{)', re.DOTALL)


    def expand(self, wikitext):
        """
        :param wikitext: the text to be expanded.

        Templates are frequently nested. Occasionally, parsing mistakes may
        cause template insertion to enter an infinite loop, for instance when
        trying to instantiate Template:Country

        {{country_{{{1}}}|{{{2}}}|{{{2}}}|size={{{size|}}}|name={{{name|}}}}}

        which is repeatedly trying to insert template 'country_', which is
        again resolved to Template:Country. The straightforward solution of
        keeping track of templates that were already inserted for the current
        article would not work, because the same template may legally be used
        more than once, with different parameters in different parts of the
        article.  Therefore, we limit the number of iterations of nested
        template inclusion.

        """
        # Test template expansion at:
        # https://en.wikipedia.org/wiki/Special:ExpandTemplates
        # https://it.wikipedia.org/wiki/Speciale:EspandiTemplate

        res = ''
        if self.frame.depth >= self.maxTemplateRecursionLevels:
            self.recursion_exceeded_1_errs += 1
            return res

        # logging.debug('%*s<expand', self.frame.depth, '')

        cur = 0
        # look for matching {{...}}
        for s, e in findMatchingBraces(wikitext, 2):
            res += wikitext[cur:s] + self.expandTemplate(wikitext[s + 2:e - 2])
            cur = e
        # leftover
        res += wikitext[cur:]
        # logging.debug('%*sexpand> %s', self.frame.depth, '', res)
        return res


    def templateParams(self, parameters):
        """
        Build a dictionary with positional or name key to expanded parameters.
        :param parameters: the parts[1:] of a template, i.e. all except the title.
        """
        templateParams = {}

        if not parameters:
            return templateParams
        # logging.debug('%*s<templateParams: %s', self.frame.length, '', '|'.join(parameters))

        # Parameters can be either named or unnamed. In the latter case, their
        # name is defined by their ordinal position (1, 2, 3, ...).

        unnamedParameterCounter = 0

        # It's legal for unnamed parameters to be skipped, in which case they
        # will get default values (if available) during actual instantiation.
        # That is {{template_name|a||c}} means parameter 1 gets
        # the value 'a', parameter 2 value is not defined, and parameter 3 gets
        # the value 'c'.  This case is correctly handled by function 'split',
        # and does not require any special handling.
        for param in parameters:
            # Spaces before or after a parameter value are normally ignored,
            # UNLESS the parameter contains a link (to prevent possible gluing
            # the link to the following text after template substitution)

            # Parameter values may contain "=" symbols, hence the parameter
            # name extends up to the first such symbol.

            # It is legal for a parameter to be specified several times, in
            # which case the last assignment takes precedence. Example:
            # "{{t|a|b|c|2=B}}" is equivalent to "{{t|a|B|c}}".
            # Therefore, we don't check if the parameter has been assigned a
            # value before, because anyway the last assignment should override
            # any previous ones.
            # FIXME: Don't use DOTALL here since parameters may be tags with
            # attributes, e.g. <div class="templatequotecite">
            # Parameters may span several lines, like:
            # {{Reflist|colwidth=30em|refs=
            # &lt;ref name=&quot;Goode&quot;&gt;Title&lt;/ref&gt;

            # The '=' might occurr within an HTML attribute:
            #   "&lt;ref name=value"
            # but we stop at first.
            m = re.match(' *([^=]*?) *?=(.*)', param, re.DOTALL)
            if m:
                # This is a named parameter.  This case also handles parameter
                # assignments like "2=xxx", where the number of an unnamed
                # parameter ("2") is specified explicitly - this is handled
                # transparently.

                parameterName = m.group(1).strip()
                parameterValue = m.group(2)

                if ']]' not in parameterValue:  # if the value does not contain a link, trim whitespace
                    parameterValue = parameterValue.strip()
                templateParams[parameterName] = parameterValue
            else:
                # this is an unnamed parameter
                unnamedParameterCounter += 1

                if ']]' not in param:  # if the value does not contain a link, trim whitespace
                    param = param.strip()
                templateParams[str(unnamedParameterCounter)] = param
        # logging.debug('%*stemplateParams> %s', self.frame.length, '', '|'.join(templateParams.values()))
        return templateParams


    def expandTemplate(self, body):
        """Expands template invocation.
        :param body: the parts of a template.

        :see http://meta.wikimedia.org/wiki/Help:Expansion for an explanation
        of the process.

        See in particular: Expansion of names and values
        http://meta.wikimedia.org/wiki/Help:Expansion#Expansion_of_names_and_values

        For most parser functions all names and values are expanded,
        regardless of what is relevant for the result. The branching functions
        (#if, #ifeq, #iferror, #ifexist, #ifexpr, #switch) are exceptions.

        All names in a template call are expanded, and the titles of the
        tplargs in the template body, after which it is determined which
        values must be expanded, and for which tplargs in the template body
        the first part (default) [sic in the original doc page].

        In the case of a tplarg, any parts beyond the first are never
        expanded.  The possible name and the value of the first part is
        expanded if the title does not match a name in the template call.

        :see code for braceSubstitution at
        https://doc.wikimedia.org/mediawiki-core/master/php/html/Parser_8php_source.html#3397:

        """

        # template        = "{{" parts "}}"

        # Templates and tplargs are decomposed in the same way, with pipes as
        # separator, even though eventually any parts in a tplarg after the first
        # (the parameter default) are ignored, and an equals sign in the first
        # part is treated as plain text.
        # Pipes inside inner templates and tplargs, or inside double rectangular
        # brackets within the template or tplargs are not taken into account in
        # this decomposition.
        # The first part is called title, the other parts are simply called parts.

        # If a part has one or more equals signs in it, the first equals sign
        # determines the division into name = value. Equals signs inside inner
        # templates and tplargs, or inside double rectangular brackets within the
        # part are not taken into account in this decomposition. Parts without
        # equals sign are indexed 1, 2, .., given as attribute in the <name> tag.

        if self.frame.depth >= self.maxTemplateRecursionLevels:
            self.recursion_exceeded_2_errs += 1
            # logging.debug('%*sEXPAND> %s', self.frame.depth, '', body)
            return ''

        logging.debug('%*sEXPAND %s', self.frame.depth, '', body)
        parts = splitParts(body)
        # title is the portion before the first |
        title = parts[0].strip()
        title = self.expand(title)

        # SUBST
        # Apply the template tag to parameters without
        # substituting into them, e.g.
        # {{subst:t|a{{{p|q}}}b}} gives the wikitext start-a{{{p|q}}}b-end
        # @see https://www.mediawiki.org/wiki/Manual:Substitution#Partial_substitution
        subst = False
        if re.match(substWords, title, re.IGNORECASE):
            title = re.sub(substWords, '', title, 1, re.IGNORECASE)
            subst = True

        if title in self.magicWords.values:
            ret = self.magicWords[title]
            logging.debug('%*s<EXPAND %s %s', self.frame.depth, '', title, ret)
            return ret

        # Parser functions.

        # For most parser functions all names and values are expanded,
        # regardless of what is relevant for the result. The branching
        # functions (#if, #ifeq, #iferror, #ifexist, #ifexpr, #switch) are
        # exceptions: for #if, #iferror, #ifexist, #ifexp, only the part that
        # is applicable is expanded; for #ifeq the first and the applicable
        # part are expanded; for #switch, expanded are the names up to and
        # including the match (or all if there is no match), and the value in
        # the case of a match or if there is no match, the default, if any.

        # The first argument is everything after the first colon.
        # It has been evaluated above.
        colon = title.find(':')
        if colon > 1:
            funct = title[:colon]
            parts[0] = title[colon + 1:].strip()  # side-effect (parts[0] not used later)
            # arguments after first are not evaluated
            ret = callParserFunction(funct, parts, self)
            logging.debug('%*s<EXPAND %s %s', self.frame.depth, '', funct, ret)
            return ret

        title = fullyQualifiedTemplateTitle(title)
        if not title:
            self.template_title_errs += 1
            return ''

        redirected = options.redirects.get(title)
        if redirected:
            title = redirected

        # get the template
        if title in options.templateCache:
            template = options.templateCache[title]
        elif title in options.templates:
            template = Template.parse(options.templates[title])
            # add it to cache
            options.templateCache[title] = template
            del options.templates[title]
        else:
            # The page being included could not be identified
            logging.debug('%*s<EXPAND %s %s', self.frame.depth, '', title, '')
            return ''

        logging.debug('%*sTEMPLATE %s: %s', self.frame.depth, '', title, template)

        # tplarg          = "{{{" parts "}}}"
        # parts           = [ title *( "|" part ) ]
        # part            = ( part-name "=" part-value ) / ( part-value )
        # part-name       = wikitext-L3
        # part-value      = wikitext-L3
        # wikitext-L3     = literal / template / tplarg / link / comment /
        #                   line-eating-comment / unclosed-comment /
        #           	    xmlish-element / *wikitext-L3

        # A tplarg may contain other parameters as well as templates, e.g.:
        #   {{{text|{{{quote|{{{1|{{error|Error: No text given}}}}}}}}}}}
        # hence no simple RE like this would work:
        #   '{{{((?:(?!{{{).)*?)}}}'
        # We must use full CF parsing.

        # the parameter name itself might be computed, e.g.:
        #   {{{appointe{{#if:{{{appointer14|}}}|r|d}}14|}}}

        # Because of the multiple uses of double-brace and triple-brace
        # syntax, expressions can sometimes be ambiguous.
        # Precedence rules specifed here:
        # http://www.mediawiki.org/wiki/Preprocessor_ABNF#Ideal_precedence
        # resolve ambiguities like this:
        #   {{{{ }}}} -> { {{{ }}} }
        #   {{{{{ }}}}} -> {{ {{{ }}} }}
        #
        # :see: https://en.wikipedia.org/wiki/Help:Template#Handling_parameters

        params = parts[1:]

        # Order of evaluation.
        # Template parameters are fully evaluated before they are passed to the template.
        # :see: https://www.mediawiki.org/wiki/Help:Templates#Order_of_evaluation
        if not subst:
            # Evaluate parameters, since they may contain templates, including
            # the symbol "=".
            # {{#ifexpr: {{{1}}} = 1 }}
            params = [self.transform(p) for p in params]

        # build a dict of name-values for the parameter values
        params = self.templateParams(params)

        # Perform parameter substitution.
        # Extend frame before subst, since there may be recursion in default
        # parameter value, e.g. {{OTRS|celebrative|date=April 2015}} in article
        # 21637542 in enwiki.
        self.frame = self.frame.push(title, params)
        instantiated = template.subst(params, self)
        value = self.transform(instantiated)
        self.frame = self.frame.pop()
        logging.debug('%*s<EXPAND %s %s', self.frame.depth, '', title, value)
        return value


# ----------------------------------------------------------------------
# parameter handling


def splitParts(paramsList):
    """
    :param paramsList: the parts of a template or tplarg.

    Split template parameters at the separator "|".
    separator "=".

    Template parameters often contain URLs, internal links, text or even
    template expressions, since we evaluate templates outside in.
    This is required for cases like:
      {{#if: {{{1}}} | {{lc:{{{1}}} | "parameter missing"}}
    Parameters are separated by "|" symbols. However, we
    cannot simply split the string on "|" symbols, since these
    also appear inside templates and internal links, e.g.

     {{if:|
      |{{#if:the president|
           |{{#if:|
               [[Category:Hatnote templates|A{{PAGENAME}}]]
            }}
       }}
     }}

    We split parts at the "|" symbols that are not inside any pair
    {{{...}}}, {{...}}, [[...]], {|...|}.
    """

    # Must consider '[' as normal in expansion of Template:EMedicine2:
    # #ifeq: ped|article|[http://emedicine.medscape.com/article/180-overview|[http://www.emedicine.com/ped/topic180.htm#{{#if: |section~}}
    # as part of:
    # {{#ifeq: ped|article|[http://emedicine.medscape.com/article/180-overview|[http://www.emedicine.com/ped/topic180.htm#{{#if: |section~}}}} ped/180{{#if: |~}}]

    # should handle both tpl arg like:
    #    4|{{{{{subst|}}}CURRENTYEAR}}
    # and tpl parameters like:
    #    ||[[Category:People|{{#if:A|A|{{PAGENAME}}}}]]

    sep = '|'
    parameters = []
    cur = 0

    for s, e in findMatchingBraces(paramsList):
        par = paramsList[cur:s].split(sep)
        if par:
            if parameters:
                # portion before | belongs to previous parameter
                parameters[-1] += par[0]
                if len(par) > 1:
                    # rest are new parameters
                    parameters.extend(par[1:])
            else:
                parameters = par
        elif not parameters:
            parameters = ['']  # create first param
        # add span to last previous parameter
        parameters[-1] += paramsList[s:e]
        cur = e
    # leftover
    par = paramsList[cur:].split(sep)
    if par:
        if parameters:
            # portion before | belongs to previous parameter
            parameters[-1] += par[0]
            if len(par) > 1:
                # rest are new parameters
                parameters.extend(par[1:])
        else:
            parameters = par

    # logging.debug('splitParts %s %s\nparams: %s', sep, paramsList, text_type(parameters))
    return parameters


def findMatchingBraces(text, ldelim=0):
    """
    :param ldelim: number of braces to match. 0 means match [[]], {{}} and {{{}}}.
    """
    # Parsing is done with respect to pairs of double braces {{..}} delimiting
    # a template, and pairs of triple braces {{{..}}} delimiting a tplarg.
    # If double opening braces are followed by triple closing braces or
    # conversely, this is taken as delimiting a template, with one left-over
    # brace outside it, taken as plain text. For any pattern of braces this
    # defines a set of templates and tplargs such that any two are either
    # separate or nested (not overlapping).

    # Unmatched double rectangular closing brackets can be in a template or
    # tplarg, but unmatched double rectangular opening brackets cannot.
    # Unmatched double or triple closing braces inside a pair of
    # double rectangular brackets are treated as plain text.
    # Other formulation: in ambiguity between template or tplarg on one hand,
    # and a link on the other hand, the structure with the rightmost opening
    # takes precedence, even if this is the opening of a link without any
    # closing, so not producing an actual link.

    # In the case of more than three opening braces the last three are assumed
    # to belong to a tplarg, unless there is no matching triple of closing
    # braces, in which case the last two opening braces are are assumed to
    # belong to a template.

    # We must skip individual { like in:
    #   {{#ifeq: {{padleft:|1|}} | { | | &nbsp;}}
    # We must resolve ambiguities like this:
    #   {{{{ }}}} -> { {{{ }}} }
    #   {{{{{ }}}}} -> {{ {{{ }}} }}
    #   {{#if:{{{{{#if:{{{nominee|}}}|nominee|candidate}}|}}}|...}}
    #   {{{!}} {{!}}}

    # Handle:
    #   {{{{{|safesubst:}}}#Invoke:String|replace|{{{1|{{{{{|safesubst:}}}PAGENAME}}}}}|%s+%([^%(]-%)$||plain=false}}
    # as well as expressions with stray }:
    #   {{{link|{{ucfirst:{{{1}}}}}} interchange}}}

    if ldelim:  # 2-3
        reOpen = re.compile('[{]{%d,}' % ldelim)  # at least ldelim
        reNext = re.compile('[{]{2,}|}{2,}')  # at least 2
    else:
        reOpen = re.compile('{{2,}|\[{2,}')
        reNext = re.compile('{{2,}|}{2,}|\[{2,}|]{2,}')  # at least 2

    cur = 0
    while True:
        m1 = reOpen.search(text, cur)
        if not m1:
            return
        lmatch = m1.end() - m1.start()
        if m1.group()[0] == '{':
            stack = [lmatch]  # stack of opening braces lengths
        else:
            stack = [-lmatch]  # negative means [
        end = m1.end()
        while True:
            m2 = reNext.search(text, end)
            if not m2:
                return  # unbalanced
            end = m2.end()
            brac = m2.group()[0]
            lmatch = m2.end() - m2.start()

            if brac == '{':
                stack.append(lmatch)
            elif brac == '}':
                while stack:
                    openCount = stack.pop()  # opening span
                    if openCount == 0:  # illegal unmatched [[
                        continue
                    if lmatch >= openCount:
                        lmatch -= openCount
                        if lmatch <= 1:  # either close or stray }
                            break
                    else:
                        # put back unmatched
                        stack.append(openCount - lmatch)
                        break
                if not stack:
                    yield m1.start(), end - lmatch
                    cur = end
                    break
                elif len(stack) == 1 and 0 < stack[0] < ldelim:
                    # ambiguous {{{{{ }}} }}
                    #yield m1.start() + stack[0], end
                    cur = end
                    break
            elif brac == '[':  # [[
                stack.append(-lmatch)
            else:  # ]]
                while stack and stack[-1] < 0:  # matching [[
                    openCount = -stack.pop()
                    if lmatch >= openCount:
                        lmatch -= openCount
                        if lmatch <= 1:  # either close or stray ]
                            break
                    else:
                        # put back unmatched (negative)
                        stack.append(lmatch - openCount)
                        break
                if not stack:
                    yield m1.start(), end - lmatch
                    cur = end
                    break
                # unmatched ]] are discarded
                cur = end


def findBalanced(text, openDelim=['[['], closeDelim=[']]']):
    """
    Assuming that text contains a properly balanced expression using
    :param openDelim: as opening delimiters and
    :param closeDelim: as closing delimiters.
    :return: an iterator producing pairs (start, end) of start and end
    positions in text containing a balanced expression.
    """
    openPat = '|'.join([re.escape(x) for x in openDelim])
    # pattern for delimiters expected after each opening delimiter
    afterPat = {o: re.compile(openPat + '|' + c, re.DOTALL) for o, c in zip(openDelim, closeDelim)}
    stack = []
    start = 0
    cur = 0
    # end = len(text)
    startSet = False
    startPat = re.compile(openPat)
    nextPat = startPat
    while True:
        next = nextPat.search(text, cur)
        if not next:
            return
        if not startSet:
            start = next.start()
            startSet = True
        delim = next.group(0)
        if delim in openDelim:
            stack.append(delim)
            nextPat = afterPat[delim]
        else:
            opening = stack.pop()
            # assert opening == openDelim[closeDelim.index(next.group(0))]
            if stack:
                nextPat = afterPat[stack[-1]]
            else:
                yield start, next.end()
                nextPat = startPat
                start = next.end()
                startSet = False
        cur = next.end()


# ----------------------------------------------------------------------
# Modules

# Only minimal support
# FIXME: import Lua modules.

def if_empty(*rest):
    """
    This implements If_empty from English Wikipedia module:

       <title>Module:If empty</title>
       <ns>828</ns>
       <text>local p = {}

    function p.main(frame)
            local args = require('Module:Arguments').getArgs(frame, {wrappers = 'Template:If empty', removeBlanks = false})

            -- For backwards compatibility reasons, the first 8 parameters can be unset instead of being blank,
            -- even though there's really no legitimate use case for this. At some point, this will be removed.
            local lowestNil = math.huge
            for i = 8,1,-1 do
                    if args[i] == nil then
                            args[i] = ''
                            lowestNil = i
                    end
            end

            for k,v in ipairs(args) do
                    if v ~= '' then
                            if lowestNil &lt; k then
                                    -- If any uses of this template depend on the behavior above, add them to a tracking category.
                                    -- This is a rather fragile, convoluted, hacky way to do it, but it ensures that this module's output won't be modified
                                    -- by it.
                                    frame:extensionTag('ref', '[[Category:Instances of Template:If_empty missing arguments]]', {group = 'TrackingCategory'})
                                    frame:extensionTag('references', '', {group = 'TrackingCategory'})
                            end
                            return v
                    end
            end
    end

    return p   </text>
    """
    for arg in rest:
        if arg:
            return arg
    return ''


# ----------------------------------------------------------------------
# String module emulation
# https://en.wikipedia.org/wiki/Module:String

def functionParams(args, vars):
    """
    Build a dictionary of var/value from :param: args.
    Parameters can be either named or unnamed. In the latter case, their
    name is taken fron :param: vars.
    """
    params = {}
    index = 1
    for var in vars:
        value = args.get(var)
        if value is None:
            value = args.get(str(index)) # positional argument
            if value is None:
                value = ''
            else:
                index += 1
        params[var] = value
    return params


def string_sub(args):
    params = functionParams(args, ('s', 'i', 'j'))
    s = params.get('s', '')
    i = int(params.get('i', 1) or 1) # or handles case of '' value
    j = int(params.get('j', -1) or -1)
    if i > 0: i -= 1             # lua is 1-based
    if j < 0: j += 1
    if j == 0: j = len(s)
    return s[i:j]


def string_sublength(args):
    params = functionParams(args, ('s', 'i', 'len'))
    s = params.get('s', '')
    i = int(params.get('i', 1) or 1) - 1 # lua is 1-based
    len = int(params.get('len', 1) or 1)
    return s[i:i+len]


def string_len(args):
    params = functionParams(args, ('s'))
    s = params.get('s', '')
    return len(s)


def string_find(args):
    params = functionParams(args, ('source', 'target', 'start', 'plain'))
    source = params.get('source', '')
    pattern = params.get('target', '')
    start = int('0'+params.get('start', 1)) - 1 # lua is 1-based
    plain = int('0'+params.get('plain', 1))
    if source == '' or pattern == '':
        return 0
    if plain:
        return source.find(pattern, start) + 1 # lua is 1-based
    else:
        return (re.compile(pattern).search(source, start) or -1) + 1


def string_pos(args):
    params = functionParams(args, ('target', 'pos'))
    target = params.get('target', '')
    pos = int(params.get('pos', 1) or 1)
    if pos > 0:
        pos -= 1 # The first character has an index value of 1
    return target[pos]


def string_replace(args):
    params = functionParams(args, ('source', 'pattern', 'replace', 'count', 'plain'))
    source = params.get('source', '')
    pattern = params.get('pattern', '')
    replace = params.get('replace', '')
    count = int(params.get('count', 0) or 0)
    plain = int(params.get('plain', 1) or 1)
    if plain:
        if count:
            return source.replace(pattern, replace, count)
        else:
            return source.replace(pattern, replace)
    else:
        return re.compile(pattern).sub(replace, source, count)


def string_rep(args):
    params = functionParams(args, ('s'))
    source = params.get('source', '')
    count = int(params.get('count', '1'))
    return source * count


# ----------------------------------------------------------------------
# Module:Roman
# http://en.wikipedia.org/w/index.php?title=Module:Roman
# Modulo:Numero_romano
# https://it.wikipedia.org/wiki/Modulo:Numero_romano

def roman_main(args):
    """Convert first arg to roman numeral if <= 5000 else :return: second arg."""
    num = int(float(args.get('1')))

    # Return a message for numbers too big to be expressed in Roman numerals.
    if 0 > num or num >= 5000:
        return args.get('2', 'N/A')

    def toRoman(n, romanNumeralMap):
        """convert integer to Roman numeral"""
        result = ""
        for integer, numeral in romanNumeralMap:
            while n >= integer:
                result += numeral
                n -= integer
        return result

    # Find the Roman numerals for numbers 4999 or less.
    smallRomans = (
        (1000, "M"),
        (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
        (90, "XC"), (50, "L"), (40, "XL"), (10, "X"),
        (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
    )
    return toRoman(num, smallRomans)

# ----------------------------------------------------------------------

modules = {
    'convert': {
        'convert': lambda x, u, *rest: x + ' ' + u,  # no conversion
    },

    'If empty': {
        'main': if_empty
    },

    'String': {
        'len': string_len,
        'sub': string_sub,
        'sublength': string_sublength,
        'pos': string_pos,
        'find': string_find,
        'replace': string_replace,
        'rep': string_rep,
    },

    'Roman': {
        'main': roman_main
    },

    'Numero romano': {
        'main': roman_main
    }
}

# ----------------------------------------------------------------------
# variables


class MagicWords(object):
    """
    One copy in each Extractor.

    @see https://doc.wikimedia.org/mediawiki-core/master/php/MagicWord_8php_source.html
    """
    names = [
        '!',
        'currentmonth',
        'currentmonth1',
        'currentmonthname',
        'currentmonthnamegen',
        'currentmonthabbrev',
        'currentday',
        'currentday2',
        'currentdayname',
        'currentyear',
        'currenttime',
        'currenthour',
        'localmonth',
        'localmonth1',
        'localmonthname',
        'localmonthnamegen',
        'localmonthabbrev',
        'localday',
        'localday2',
        'localdayname',
        'localyear',
        'localtime',
        'localhour',
        'numberofarticles',
        'numberoffiles',
        'numberofedits',
        'articlepath',
        'pageid',
        'sitename',
        'server',
        'servername',
        'scriptpath',
        'stylepath',
        'pagename',
        'pagenamee',
        'fullpagename',
        'fullpagenamee',
        'namespace',
        'namespacee',
        'namespacenumber',
        'currentweek',
        'currentdow',
        'localweek',
        'localdow',
        'revisionid',
        'revisionday',
        'revisionday2',
        'revisionmonth',
        'revisionmonth1',
        'revisionyear',
        'revisiontimestamp',
        'revisionuser',
        'revisionsize',
        'subpagename',
        'subpagenamee',
        'talkspace',
        'talkspacee',
        'subjectspace',
        'subjectspacee',
        'talkpagename',
        'talkpagenamee',
        'subjectpagename',
        'subjectpagenamee',
        'numberofusers',
        'numberofactiveusers',
        'numberofpages',
        'currentversion',
        'rootpagename',
        'rootpagenamee',
        'basepagename',
        'basepagenamee',
        'currenttimestamp',
        'localtimestamp',
        'directionmark',
        'contentlanguage',
        'numberofadmins',
        'cascadingsources',
    ]

    def __init__(self):
        self.values = {'!': '|'}

    def __getitem__(self, name):
        return self.values.get(name)

    def __setitem__(self, name, value):
        self.values[name] = value

    switches = (
        '__NOTOC__',
        '__FORCETOC__',
        '__TOC__',
        '__TOC__',
        '__NEWSECTIONLINK__',
        '__NONEWSECTIONLINK__',
        '__NOGALLERY__',
        '__HIDDENCAT__',
        '__NOCONTENTCONVERT__',
        '__NOCC__',
        '__NOTITLECONVERT__',
        '__NOTC__',
        '__START__',
        '__END__',
        '__INDEX__',
        '__NOINDEX__',
        '__STATICREDIRECT__',
        '__DISAMBIG__'
    )


magicWordsRE = re.compile('|'.join(MagicWords.switches))


# ----------------------------------------------------------------------
# parser functions utilities


def ucfirst(string):
    """:return: a string with just its first character uppercase
    We can't use title() since it coverts all words.
    """
    if string:
        return string[0].upper() + string[1:]
    else:
        return ''


def lcfirst(string):
    """:return: a string with its first character lowercase"""
    if string:
        if len(string) > 1:
            return string[0].lower() + string[1:]
        else:
            return string.lower()
    else:
        return ''


def fullyQualifiedTemplateTitle(templateTitle):
    """
    Determine the namespace of the page being included through the template
    mechanism
    """
    if templateTitle.startswith(':'):
        # Leading colon by itself implies main namespace, so strip this colon
        return ucfirst(templateTitle[1:])
    else:
        m = re.match('([^:]*)(:.*)', templateTitle)
        if m:
            # colon found but not in the first position - check if it
            # designates a known namespace
            prefix = normalizeNamespace(m.group(1))
            if prefix in options.knownNamespaces:
                return prefix + ucfirst(m.group(2))
    # The title of the page being included is NOT in the main namespace and
    # lacks any other explicit designation of the namespace - therefore, it
    # is resolved to the Template namespace (that's the default for the
    # template inclusion mechanism).

    # This is a defense against pages whose title only contains UTF-8 chars
    # that are reduced to an empty string. Right now I can think of one such
    # case - <C2><A0> which represents the non-breaking space.
    # In this particular case, this page is a redirect to [[Non-nreaking
    # space]], but having in the system a redirect page with an empty title
    # causes numerous problems, so we'll live happier without it.
    if templateTitle:
        return options.templatePrefix + ucfirst(templateTitle)
    else:
        return ''  # caller may log as error


def normalizeNamespace(ns):
    return ucfirst(ns)


# ----------------------------------------------------------------------
# Parser functions
# see http://www.mediawiki.org/wiki/Help:Extension:ParserFunctions
# https://github.com/Wikia/app/blob/dev/extensions/ParserFunctions/ParserFunctions_body.php


class Infix:
    """Infix operators.
    The calling sequence for the infix is:
      x |op| y
    """

    def __init__(self, function):
        self.function = function

    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __or__(self, other):
        return self.function(other)

    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))

    def __rshift__(self, other):
        return self.function(other)

    def __call__(self, value1, value2):
        return self.function(value1, value2)


ROUND = Infix(lambda x, y: round(x, y))


from math import floor, ceil, pi, e, trunc, exp, log as ln, sin, cos, tan, asin, acos, atan


def sharp_expr(extr, expr):
    """Tries converting a lua expr into a Python expr."""
    try:
        expr = extr.expand(expr)
        expr = re.sub('(?<![!<>])=', '==', expr) # negative lookbehind
        expr = re.sub('mod', '%', expr)          # no \b here
        expr = re.sub('\bdiv\b', '/', expr)
        expr = re.sub('\bround\b', '|ROUND|', expr)
        return text_type(eval(expr))
    except:
        return '<span class="error">%s</span>' % expr


def sharp_if(extr, testValue, valueIfTrue, valueIfFalse=None, *args):
    # In theory, we should evaluate the first argument here,
    # but it was evaluated while evaluating part[0] in expandTemplate().
    if testValue.strip():
        # The {{#if:}} function is an if-then-else construct.
        # The applied condition is: "The condition string is non-empty".
        valueIfTrue = extr.expand(valueIfTrue.strip()) # eval
        if valueIfTrue:
            return valueIfTrue
    elif valueIfFalse:
        return extr.expand(valueIfFalse.strip()) # eval
    return ""


def sharp_ifeq(extr, lvalue, rvalue, valueIfTrue, valueIfFalse=None, *args):
    rvalue = rvalue.strip()
    if rvalue:
        # lvalue is always evaluated
        if lvalue.strip() == rvalue:
            # The {{#ifeq:}} function is an if-then-else construct. The
            # applied condition is "is rvalue equal to lvalue". Note that this
            # does only string comparison while MediaWiki implementation also
            # supports numerical comparissons.

            if valueIfTrue:
                return extr.expand(valueIfTrue.strip())
        else:
            if valueIfFalse:
                return extr.expand(valueIfFalse.strip())
    return ""


def sharp_iferror(extr, test, then='', Else=None, *args):
    if re.match('<(?:strong|span|p|div)\s(?:[^\s>]*\s+)*?class="(?:[^"\s>]*\s+)*?error(?:\s[^">]*)?"', test):
        return extr.expand(then.strip())
    elif Else is None:
        return test.strip()
    else:
        return extr.expand(Else.strip())


def sharp_switch(extr, primary, *params):
    # FIXME: we don't support numeric expressions in primary

    # {{#switch: comparison string
    #  | case1 = result1
    #  | case2
    #  | case4 = result2
    #  | 1 | case5 = result3
    #  | #default = result4
    # }}

    primary = primary.strip()
    found = False  # for fall through cases
    default = None
    rvalue = None
    lvalue = ''
    for param in params:
        # handle cases like:
        #  #default = [http://www.perseus.tufts.edu/hopper/text?doc=Perseus...]
        pair = param.split('=', 1)
        lvalue = extr.expand(pair[0].strip())
        rvalue = None
        if len(pair) > 1:
            # got "="
            rvalue = extr.expand(pair[1].strip())
            # check for any of multiple values pipe separated
            if found or primary in [v.strip() for v in lvalue.split('|')]:
                # Found a match, return now
                return rvalue
            elif lvalue == '#default':
                default = rvalue
            rvalue = None  # avoid defaulting to last case
        elif lvalue == primary:
            # If the value matches, set a flag and continue
            found = True
    # Default case
    # Check if the last item had no = sign, thus specifying the default case
    if rvalue is not None:
        return lvalue
    elif default is not None:
        return default
    return ''


# Extension Scribunto: https://www.mediawiki.org/wiki/Extension:Scribunto
def sharp_invoke(module, function, args):
    functions = modules.get(module)
    if functions:
        funct = functions.get(function)
        if funct:
            return text_type(funct(args))
    return ''


parserFunctions = {

    '#expr': sharp_expr,

    '#if': sharp_if,

    '#ifeq': sharp_ifeq,

    '#iferror': sharp_iferror,

    '#ifexpr': lambda *args: '',  # not supported

    '#ifexist': lambda extr, title, ifex, ifnex: extr.expand(ifnex), # assuming title is not present

    '#rel2abs': lambda *args: '',  # not supported

    '#switch': sharp_switch,

    '#language': lambda *args: '', # not supported

    '#time': lambda *args: '',     # not supported

    '#timel': lambda *args: '',    # not supported

    '#titleparts': lambda *args: '', # not supported

    # This function is used in some pages to construct links
    # http://meta.wikimedia.org/wiki/Help:URL
    'urlencode': lambda extr, string, *rest: quote(string.encode('utf-8')),

    'lc': lambda extr, string, *rest: string.lower() if string else '',

    'lcfirst': lambda extr, string, *rest: lcfirst(string),

    'uc': lambda extr, string, *rest: string.upper() if string else '',

    'ucfirst': lambda extr, string, *rest: ucfirst(string),

    'int': lambda extr, string, *rest: text_type(int(string)),

}


def callParserFunction(functionName, args, extractor):
    """
    Parser functions have similar syntax as templates, except that
    the first argument is everything after the first colon.
    :return: the result of the invocation, None in case of failure.

    :param: args not yet expanded (see branching functions).
    https://www.mediawiki.org/wiki/Help:Extension:ParserFunctions
    """

    try:
        # https://it.wikipedia.org/wiki/Template:Str_endswith has #Invoke
        functionName = functionName.lower()
        if functionName == '#invoke':
            module, fun = args[0].strip(), args[1].strip()
            logging.debug('%*s#invoke %s %s %s', extractor.frame.depth, '', module, fun, args[2:])
            # special handling of frame
            if len(args) == 2:
                # find parameters in frame whose title is the one of the original
                # template invocation
                templateTitle = fullyQualifiedTemplateTitle(module)
                if not templateTitle:
                    logging.warn("Template with empty title")
                params = None
                frame = extractor.frame
                while frame:
                    if frame.title == templateTitle:
                        params = frame.args
                        break
                    frame = frame.prev
            else:
                params = [extractor.transform(p) for p in args[2:]] # evaluates them
                params = extractor.templateParams(params)
            ret = sharp_invoke(module, fun, params)
            logging.debug('%*s<#invoke %s %s %s', extractor.frame.depth, '', module, fun, ret)
            return ret
        if functionName in parserFunctions:
            # branching functions use the extractor to selectively evaluate args
            return parserFunctions[functionName](extractor, *args)
    except:
        return ""  # FIXME: fix errors
    return ""


# ----------------------------------------------------------------------
# Expand using WikiMedia API
# import json

# def expand(text):
#     """Expand templates invoking MediaWiki API"""
#     text = urlib.urlencodew(text.encode('utf-8'))
#     base = urlbase[:urlbase.rfind('/')]
#     url = base + "/w/api.php?action=expandtemplates&format=json&text=" + text
#     exp = json.loads(urllib.urlopen(url))
#     return exp['expandtemplates']['*']

# ----------------------------------------------------------------------
# Extract Template definition

reNoinclude = re.compile(r'<noinclude>(?:.*?)</noinclude>', re.DOTALL)
reIncludeonly = re.compile(r'<includeonly>|</includeonly>', re.DOTALL)

def define_template(title, page):
    """
    Adds a template defined in the :param page:.
    @see https://en.wikipedia.org/wiki/Help:Template#Noinclude.2C_includeonly.2C_and_onlyinclude
    """
    # title = normalizeTitle(title)

    # sanity check (empty template, e.g. Template:Crude Oil Prices))
    if not page: return

    # check for redirects
    m = re.match('#REDIRECT.*?\[\[([^\]]*)]]', page[0], re.IGNORECASE)
    if m:
        options.redirects[title] = m.group(1)  # normalizeTitle(m.group(1))
        return

    text = unescape(''.join(page))

    # We're storing template text for future inclusion, therefore,
    # remove all <noinclude> text and keep all <includeonly> text
    # (but eliminate <includeonly> tags per se).
    # However, if <onlyinclude> ... </onlyinclude> parts are present,
    # then only keep them and discard the rest of the template body.
    # This is because using <onlyinclude> on a text fragment is
    # equivalent to enclosing it in <includeonly> tags **AND**
    # enclosing all the rest of the template body in <noinclude> tags.

    # remove comments
    text = comment.sub('', text)

    # eliminate <noinclude> fragments
    text = reNoinclude.sub('', text)
    # eliminate unterminated <noinclude> elements
    text = re.sub(r'<noinclude\s*>.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'<noinclude/>', '', text)

    onlyincludeAccumulator = ''
    for m in re.finditer('<onlyinclude>(.*?)</onlyinclude>', text, re.DOTALL):
        onlyincludeAccumulator += m.group(1)
    if onlyincludeAccumulator:
        text = onlyincludeAccumulator
    else:
        text = reIncludeonly.sub('', text)

    if text:
        if title in options.templates:
            logging.warn('Redefining: %s', title)
        options.templates[title] = text


# ----------------------------------------------------------------------

def dropNested(text, openDelim, closeDelim):
    """
    A matching function for nested expressions, e.g. namespaces and tables.
    """
    openRE = re.compile(openDelim, re.IGNORECASE)
    closeRE = re.compile(closeDelim, re.IGNORECASE)
    # partition text in separate blocks { } { }
    spans = []                  # pairs (s, e) for each partition
    nest = 0                    # nesting level
    start = openRE.search(text, 0)
    if not start:
        return text
    end = closeRE.search(text, start.end())
    next = start
    while end:
        next = openRE.search(text, next.end())
        if not next:            # termination
            while nest:         # close all pending
                nest -= 1
                end0 = closeRE.search(text, end.end())
                if end0:
                    end = end0
                else:
                    break
            spans.append((start.start(), end.end()))
            break
        while end.end() < next.start():
            # { } {
            if nest:
                nest -= 1
                # try closing more
                last = end.end()
                end = closeRE.search(text, end.end())
                if not end:     # unbalanced
                    if spans:
                        span = (spans[0][0], last)
                    else:
                        span = (start.start(), last)
                    spans = [span]
                    break
            else:
                spans.append((start.start(), end.end()))
                # advance start, find next close
                start = next
                end = closeRE.search(text, next.end())
                break           # { }
        if next != start:
            # { { }
            nest += 1
    # collect text outside partitions
    return dropSpans(spans, text)


def dropSpans(spans, text):
    """
    Drop from text the blocks identified in :param spans:, possibly nested.
    """
    spans.sort()
    res = ''
    offset = 0
    for s, e in spans:
        if offset <= s:         # handle nesting
            if offset < s:
                res += text[offset:s]
            offset = e
    res += text[offset:]
    return res


# ----------------------------------------------------------------------
# WikiLinks

# May be nested [[File:..|..[[..]]..|..]], [[Category:...]], etc.
# Also: [[Help:IPA for Catalan|[andora]]]


def replaceInternalLinks(text):
    """
    Replaces internal links of the form:
    [[title |...|label]]trail

    with title concatenated with trail, when present, e.g. 's' for plural.

    See https://www.mediawiki.org/wiki/Help:Links#Internal_links
    """
    # call this after removal of external links, so we need not worry about
    # triple closing ]]].
    cur = 0
    res = ''
    for s, e in findBalanced(text):
        m = tailRE.match(text, e)
        if m:
            trail = m.group(0)
            end = m.end()
        else:
            trail = ''
            end = e
        inner = text[s + 2:e - 2]
        # find first |
        pipe = inner.find('|')
        if pipe < 0:
            title = inner
            label = title
        else:
            title = inner[:pipe].rstrip()
            # find last |
            curp = pipe + 1
            for s1, e1 in findBalanced(inner):
                last = inner.rfind('|', curp, s1)
                if last >= 0:
                    pipe = last  # advance
                curp = e1
            label = inner[pipe + 1:].strip()
        res += text[cur:s] + makeInternalLink(title, label) + trail
        cur = end
    return res + text[cur:]


# the official version is a method in class Parser, similar to this:
# def replaceInternalLinks2(text):
#     global wgExtraInterlanguageLinkPrefixes

#     # the % is needed to support urlencoded titles as well
#     tc = Title::legalChars() + '#%'
#     # Match a link having the form [[namespace:link|alternate]]trail
#     e1 = re.compile("([%s]+)(?:\\|(.+?))?]](.*)" % tc, re.S | re.D)
#     # Match cases where there is no "]]", which might still be images
#     e1_img = re.compile("([%s]+)\\|(.*)" % tc, re.S | re.D)

#     holders = LinkHolderArray(self)

#     # split the entire text string on occurrences of [[
#     iterBrackets = re.compile('[[').finditer(text)

#     m in iterBrackets.next()
#     # get the first element (all text up to first [[)
#     s = text[:m.start()]
#     cur = m.end()

#     line = s

#     useLinkPrefixExtension = self.getTargetLanguage().linkPrefixExtension()
#     e2 = None
#     if useLinkPrefixExtension:
#         # Match the end of a line for a word that is not followed by whitespace,
#         # e.g. in the case of "The Arab al[[Razi]]",  "al" will be matched
#         global wgContLang
#         charset = wgContLang.linkPrefixCharset()
#         e2 = re.compile("((?>.*[^charset]|))(.+)", re.S | re.D | re.U)

#     if self.mTitle is None:
#         raise MWException(__METHOD__ + ": \self.mTitle is null\n")

#     nottalk = not self.mTitle.isTalkPage()

#     if useLinkPrefixExtension:
#         m = e2.match(s)
#         if m:
#             first_prefix = m.group(2)
#         else:
#             first_prefix = false
#     else:
#         prefix = ''

#     useSubpages = self.areSubpagesAllowed()

#     for m in iterBrackets:
#         line = text[cur:m.start()]
#         cur = m.end()

#         # TODO: Check for excessive memory usage

#         if useLinkPrefixExtension:
#             m = e2.match(e2)
#             if m:
#                 prefix = m.group(2)
#                 s = m.group(1)
#             else:
#                 prefix = ''
#             # first link
#             if first_prefix:
#                 prefix = first_prefix
#                 first_prefix = False

#         might_be_img = False

#         m = e1.match(line)
#         if m: # page with normal label or alt
#             label = m.group(2)
#             # If we get a ] at the beginning of m.group(3) that means we have a link that is something like:
#             # [[Image:Foo.jpg|[http://example.com desc]]] <- having three ] in a row fucks up,
#             # the real problem is with the e1 regex
#             # See bug 1300.
#             #
#             # Still some problems for cases where the ] is meant to be outside punctuation,
#             # and no image is in sight. See bug 2095.
#             #
#             if label and m.group(3)[0] == ']' and '[' in label:
#                 label += ']' # so that replaceExternalLinks(label) works later
#                 m.group(3) = m.group(3)[1:]
#             # fix up urlencoded title texts
#             if '%' in m.group(1):
#                 # Should anchors '#' also be rejected?
#                 m.group(1) = str_replace(array('<', '>'), array('&lt', '&gt'), rawurldecode(m.group(1)))
#             trail = m.group(3)
#         else:
#             m = e1_img.match(line):
#             if m:
#                 # Invalid, but might be an image with a link in its caption
#                 might_be_img = true
#                 label = m.group(2)
#                 if '%' in m.group(1):
#                     m.group(1) = rawurldecode(m.group(1))
#                 trail = ""
#             else:		# Invalid form; output directly
#                 s += prefix + '[[' + line
#                 continue

#         origLink = m.group(1)

#         # Dont allow internal links to pages containing
#         # PROTO: where PROTO is a valid URL protocol these
#         # should be external links.
#         if (preg_match('/^(?i:' + self.mUrlProtocols + ')/', origLink)) {
#             s += prefix + '[[' + line
#             continue
#         }

#         # Make subpage if necessary
#         if useSubpages:
#             link = self.maybeDoSubpageLink(origLink, label)
#         else:
#             link = origLink

#         noforce = origLink[0] != ':'
#         if not noforce:
#             # Strip off leading ':'
#             link = link[1:]

#         nt = Title::newFromText(self.mStripState.unstripNoWiki(link))
#         if nt is None:
#             s += prefix + '[[' + line
#             continue

#         ns = nt.getNamespace()
#         iw = nt.getInterwiki()

#         if might_be_img {	# if this is actually an invalid link
#             if (ns == NS_FILE and noforce) { # but might be an image
#                 found = False
#                 while True:
#                     # look at the next 'line' to see if we can close it there
#                     next_line = iterBrakets.next()
#                     if not next_line:
#                         break
#                     m = explode(']]', next_line, 3)
#                     if m.lastindex == 3:
#                         # the first ]] closes the inner link, the second the image
#                         found = True
#                         label += "[[%s]]%s" % (m.group(0), m.group(1))
#                         trail = m.group(2)
#                         break
#                     elif m.lastindex == 2:
#                         # if there is exactly one ]] that is fine, we will keep looking
#                         label += "[[{m[0]}]]{m.group(1)}"
#                     else:
#                         # if next_line is invalid too, we need look no further
#                         label += '[[' + next_line
#                         break
#                 if not found:
#                     # we couldnt find the end of this imageLink, so output it raw
#                     # but dont ignore what might be perfectly normal links in the text we ve examined
#                     holders.merge(self.replaceInternalLinks2(label))
#                     s += "{prefix}[[%s|%s" % (link, text)
#                     # note: no trail, because without an end, there *is* no trail
#                     continue
#             } else: # it is not an image, so output it raw
#                 s += "{prefix}[[%s|%s" % (link, text)
#                 # note: no trail, because without an end, there *is* no trail
#                      continue
#         }

#         wasblank = (text == '')
#         if wasblank:
#             text = link
#         else:
#             # Bug 4598 madness.  Handle the quotes only if they come from the alternate part
#             # [[Lista d''e paise d''o munno]] . <a href="...">Lista d''e paise d''o munno</a>
#             # [[Criticism of Harry Potter|Criticism of ''Harry Potter'']]
#             #    . <a href="Criticism of Harry Potter">Criticism of <i>Harry Potter</i></a>
#             text = self.doQuotes(text)

#         # Link not escaped by : , create the various objects
#         if noforce and not nt.wasLocalInterwiki():
#             # Interwikis
#             if iw and mOptions.getInterwikiMagic() and nottalk and (
#                     Language::fetchLanguageName(iw, None, 'mw') or
#                     in_array(iw, wgExtraInterlanguageLinkPrefixes)):
#                 # Bug 24502: filter duplicates
#                 if iw not in mLangLinkLanguages:
#                     self.mLangLinkLanguages[iw] = True
#                     self.mOutput.addLanguageLink(nt.getFullText())

#                 s = rstrip(s + prefix)
#                 s += strip(trail, "\n") == '' ? '': prefix + trail
#                 continue

#             if ns == NS_FILE:
#                 if not wfIsBadImage(nt.getDBkey(), self.mTitle):
#                     if wasblank:
#                         # if no parameters were passed, text
#                         # becomes something like "File:Foo.png",
#                         # which we dont want to pass on to the
#                         # image generator
#                         text = ''
#                     else:
#                         # recursively parse links inside the image caption
#                         # actually, this will parse them in any other parameters, too,
#                         # but it might be hard to fix that, and it doesnt matter ATM
#                         text = self.replaceExternalLinks(text)
#                         holders.merge(self.replaceInternalLinks2(text))
#                     # cloak any absolute URLs inside the image markup, so replaceExternalLinks() wont touch them
#                     s += prefix + self.armorLinks(
#                         self.makeImage(nt, text, holders)) + trail
#                 else:
#                     s += prefix + trail
#                 continue

#             if ns == NS_CATEGORY:
#                 s = rstrip(s + "\n") # bug 87

#                 if wasblank:
#                     sortkey = self.getDefaultSort()
#                 else:
#                     sortkey = text
#                 sortkey = Sanitizer::decodeCharReferences(sortkey)
#                 sortkey = str_replace("\n", '', sortkey)
#                 sortkey = self.getConverterLanguage().convertCategoryKey(sortkey)
#                 self.mOutput.addCategory(nt.getDBkey(), sortkey)

#                 s += strip(prefix + trail, "\n") == '' ? '' : prefix + trail

#                 continue
#             }
#         }

#         # Self-link checking. For some languages, variants of the title are checked in
#         # LinkHolderArray::doVariants() to allow batching the existence checks necessary
#         # for linking to a different variant.
#         if ns != NS_SPECIAL and nt.equals(self.mTitle) and !nt.hasFragment():
#             s += prefix + Linker::makeSelfLinkObj(nt, text, '', trail)
#                  continue

#         # NS_MEDIA is a pseudo-namespace for linking directly to a file
#         # @todo FIXME: Should do batch file existence checks, see comment below
#         if ns == NS_MEDIA:
#             # Give extensions a chance to select the file revision for us
#             options = []
#             descQuery = False
#             Hooks::run('BeforeParserFetchFileAndTitle',
#                        [this, nt, &options, &descQuery])
#             # Fetch and register the file (file title may be different via hooks)
#             file, nt = self.fetchFileAndTitle(nt, options)
#             # Cloak with NOPARSE to avoid replacement in replaceExternalLinks
#             s += prefix + self.armorLinks(
#                 Linker::makeMediaLinkFile(nt, file, text)) + trail
#             continue

#         # Some titles, such as valid special pages or files in foreign repos, should
#         # be shown as bluelinks even though they are not included in the page table
#         #
#         # @todo FIXME: isAlwaysKnown() can be expensive for file links; we should really do
#         # batch file existence checks for NS_FILE and NS_MEDIA
#         if iw == '' and nt.isAlwaysKnown():
#             self.mOutput.addLink(nt)
#             s += self.makeKnownLinkHolder(nt, text, array(), trail, prefix)
#         else:
#             # Links will be added to the output link list after checking
#             s += holders.makeHolder(nt, text, array(), trail, prefix)
#     }
#     return holders


def makeInternalLink(title, label):
    colon = title.find(':')
    if colon > 0 and title[:colon] not in options.acceptedNamespaces:
        return ''
    if colon == 0:
        # drop also :File:
        colon2 = title.find(':', colon + 1)
        if colon2 > 1 and title[colon + 1:colon2] not in options.acceptedNamespaces:
            return ''
    if options.keepLinks:
        return '<a href="%s">%s</a>' % (quote(title.encode('utf-8')), label)
    else:
        return label


# ----------------------------------------------------------------------
# External links

# from: https://doc.wikimedia.org/mediawiki-core/master/php/DefaultSettings_8php_source.html

wgUrlProtocols = [
    'bitcoin:', 'ftp://', 'ftps://', 'geo:', 'git://', 'gopher://', 'http://',
    'https://', 'irc://', 'ircs://', 'magnet:', 'mailto:', 'mms://', 'news:',
    'nntp://', 'redis://', 'sftp://', 'sip:', 'sips:', 'sms:', 'ssh://',
    'svn://', 'tel:', 'telnet://', 'urn:', 'worldwind://', 'xmpp:', '//'
]

# from: https://doc.wikimedia.org/mediawiki-core/master/php/Parser_8php_source.html

# Constants needed for external link processing
# Everything except bracket, space, or control characters
# \p{Zs} is unicode 'separator, space' category. It covers the space 0x20
# as well as U+3000 is IDEOGRAPHIC SPACE for bug 19052
EXT_LINK_URL_CLASS = r'[^][<>"\x00-\x20\x7F\s]'
ANCHOR_CLASS = r'[^][\x00-\x08\x0a-\x1F]'
ExtLinkBracketedRegex = re.compile(
    '\[(((?i)' + '|'.join(wgUrlProtocols) + ')' + EXT_LINK_URL_CLASS + r'+)' +
    r'\s*((?:' + ANCHOR_CLASS + r'|\[\[' + ANCHOR_CLASS + r'+\]\])' + r'*?)\]',
    re.S | re.U)
# A simpler alternative:
# ExtLinkBracketedRegex = re.compile(r'\[(.*?)\](?!])')

EXT_IMAGE_REGEX = re.compile(
    r"""^(http://|https://)([^][<>"\x00-\x20\x7F\s]+)
    /([A-Za-z0-9_.,~%\-+&;#*?!=()@\x80-\xFF]+)\.((?i)gif|png|jpg|jpeg)$""",
    re.X | re.S | re.U)


def replaceExternalLinks(text):
    """
    https://www.mediawiki.org/wiki/Help:Links#External_links
    [URL anchor text]
    """
    s = ''
    cur = 0
    for m in ExtLinkBracketedRegex.finditer(text):
        s += text[cur:m.start()]
        cur = m.end()

        url = m.group(1)
        label = m.group(3)

        # # The characters '<' and '>' (which were escaped by
        # # removeHTMLtags()) should not be included in
        # # URLs, per RFC 2396.
        # m2 = re.search('&(lt|gt);', url)
        # if m2:
        #     link = url[m2.end():] + ' ' + link
        #     url = url[0:m2.end()]

        # If the link text is an image URL, replace it with an <img> tag
        # This happened by accident in the original parser, but some people used it extensively
        m = EXT_IMAGE_REGEX.match(label)
        if m:
            label = makeExternalImage(label)

        # Use the encoded URL
        # This means that users can paste URLs directly into the text
        # Funny characters like ö aren't valid in URLs anyway
        # This was changed in August 2004
        s += makeExternalLink(url, label)  # + trail

    return s + text[cur:]


def makeExternalLink(url, anchor):
    """Function applied to wikiLinks"""
    if options.keepLinks:
        return '<a href="%s">%s</a>' % (quote(url.encode('utf-8')), anchor)
    else:
        return anchor


def makeExternalImage(url, alt=''):
    if options.keepLinks:
        return '<img src="%s" alt="%s">' % (url, alt)
    else:
        return alt


# ----------------------------------------------------------------------

# match tail after wikilink
tailRE = re.compile('\w+')

syntaxhighlight = re.compile('&lt;syntaxhighlight .*?&gt;(.*?)&lt;/syntaxhighlight&gt;', re.DOTALL)

# skip level 1, it is page name level
section = re.compile(r'(==+)\s*(.*?)\s*\1')

listOpen = {'*': '<ul>', '#': '<ol>', ';': '<dl>', ':': '<dl>'}
listClose = {'*': '</ul>', '#': '</ol>', ';': '</dl>', ':': '</dl>'}
listItem = {'*': '<li>%s</li>', '#': '<li>%s</<li>', ';': '<dt>%s</dt>',
            ':': '<dd>%s</dd>'}


def compact(text):
    """Deal with headers, lists, empty sections, residuals of tables.
    :param text: convert to HTML.
    """

    page = []             # list of paragraph
    headers = {}          # Headers for unfilled sections
    emptySection = False  # empty sections are discarded
    listLevel = []        # nesting of lists
    listCount = []        # count of each list (it should be always in the same length of listLevel)
    for line in text.split('\n'):
        if not line:            # collapse empty lines
            # if there is an opening list, close it if we see an empty line
            if len(listLevel):
                page.append(line)
                if options.toHTML:
                    for c in reversed(listLevel):
                        page.append(listClose[c])
                listLevel = []
                listCount = []
                emptySection = False
            elif page and page[-1]:
                page.append('')
            continue
        # Handle section titles
        m = section.match(line)
        if m:
            title = m.group(2)
            lev = len(m.group(1)) # header level
            if options.toHTML:
                page.append("<h%d>%s</h%d>" % (lev, title, lev))
            if title and title[-1] not in '!?':
                title += '.'    # terminate sentence.
            headers[lev] = title
            # drop previous headers
            for i in list(headers.keys()):
                if i > lev:
                    del headers[i]
            emptySection = True
            listLevel = []
            listCount = []
            continue
        # Handle page title
        elif line.startswith('++'):
            title = line[2:-2]
            if title:
                if title[-1] not in '!?':
                    title += '.'
                page.append(title)
        # handle indents
        elif line[0] == ':':
            # page.append(line.lstrip(':*#;'))
            continue
        # handle lists
        elif line[0] in '*#;:':
            i = 0
            # c: current level char
            # n: next level char
            for c, n in zip_longest(listLevel, line, fillvalue=''):
                if not n or n not in '*#;:': # shorter or different
                    if c:
                        if options.toHTML:
                            page.append(listClose[c])
                        listLevel = listLevel[:-1]
                        listCount = listCount[:-1]
                        continue
                    else:
                        break
                # n != ''
                if c != n and (not c or (c not in ';:' and n not in ';:')):
                    if c:
                        # close level
                        if options.toHTML:
                            page.append(listClose[c])
                        listLevel = listLevel[:-1]
                        listCount = listCount[:-1]
                    listLevel += n
                    listCount.append(0)
                    if options.toHTML:
                        page.append(listOpen[n])
                i += 1
            n = line[i - 1]  # last list char
            line = line[i:].strip()
            if line:  # FIXME: n is '"'
                if options.keepLists:
                    if options.keepSections:
                        # emit open sections
                        items = sorted(headers.items())
                        for _, v in items:
                            page.append("Section::::" + v)
                    headers.clear()
                    # use item count for #-lines
                    listCount[i - 1] += 1
                    bullet = 'BULLET::::%d. ' % listCount[i - 1] if n == '#' else 'BULLET::::- '
                    page.append('{0:{1}s}'.format(bullet, len(listLevel)) + line)
                elif options.toHTML:
                    if n not in listItem: 
                        n = '*'
                    page.append(listItem[n] % line)
        elif len(listLevel):
            if options.toHTML:
                for c in reversed(listLevel):
                    page.append(listClose[c])
            listLevel = []
            listCount = []
            page.append(line)

        # Drop residuals of lists
        elif line[0] in '{|' or line[-1] == '}':
            continue
        # Drop irrelevant lines
        elif (line[0] == '(' and line[-1] == ')') or line.strip('.-') == '':
            continue
        elif len(headers):
            if options.keepSections:
                items = sorted(headers.items())
                for i, v in items:
                    page.append("Section::::" + v)
            headers.clear()
            page.append(line)  # first line
            emptySection = False
        elif not emptySection:
            # Drop preformatted
            if line[0] != ' ':  # dangerous
                page.append(line)
    return page


def handle_unicode(entity):
    numeric_code = int(entity[2:-1])
    if numeric_code >= 0x10000: return ''
    return chr(numeric_code)


# ------------------------------------------------------------------------------
# Output


class NextFile(object):
    """
    Synchronous generation of next available file name.
    """

    filesPerDir = 100

    def __init__(self, path_name):
        self.path_name = path_name
        self.dir_index = -1
        self.file_index = -1

    def __next__(self):
        self.file_index = (self.file_index + 1) % NextFile.filesPerDir
        if self.file_index == 0:
            self.dir_index += 1
        dirname = self._dirname()
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        return self._filepath()

    next = __next__

    def _dirname(self):
        char1 = self.dir_index % 26
        char2 = self.dir_index // 26 % 26
        return os.path.join(self.path_name, '%c%c' % (ord('A') + char2, ord('A') + char1))

    def _filepath(self):
        return '%s/wiki_%02d' % (self._dirname(), self.file_index)


class OutputSplitter(object):
    """
    File-like object, that splits output to multiple files of a given max size.
    """

    def __init__(self, nextFile, max_file_size=0, compress=True):
        """
        :param nextFile: a NextFile object from which to obtain filenames
            to use.
        :param max_file_size: the maximum size of each file.
        :para compress: whether to write data with bzip compression.
        """
        self.nextFile = nextFile
        self.compress = compress
        self.max_file_size = max_file_size
        self.file = self.open(next(self.nextFile))

    def reserve(self, size):
        if self.file.tell() + size > self.max_file_size:
            self.close()
            self.file = self.open(next(self.nextFile))

    def write(self, data):
        self.reserve(len(data))
        self.file.write(data)

    def close(self):
        self.file.close()

    def open(self, filename):
        if self.compress:
            return bz2.BZ2File(filename + '.bz2', 'w')
        else:
            return open(filename, 'wb')


# ----------------------------------------------------------------------
# READER

tagRE = re.compile(r'(.*?)<(/?\w+)[^>]*?>(?:([^<]*)(<.*?>)?)?')
#                    1     2               3      4
keyRE = re.compile(r'key="(\d*)"')
catRE = re.compile(r'\[\[Category:([^\|]+).*\]\].*')  # capture the category name [[Category:Category name|Sortkey]]"

def load_templates(file, output_file=None):
    """
    Load templates from :param file:.
    :param output_file: file where to save templates and modules.
    """
    options.templatePrefix = options.templateNamespace + ':'
    options.modulePrefix = options.moduleNamespace + ':'

    if output_file:
        output = codecs.open(output_file, 'wb', 'utf-8')
    for page_count, page_data in enumerate(pages_from(file)):
        id, revid, title, ns,catSet, page = page_data
        if not output_file and (not options.templateNamespace or
                                not options.moduleNamespace):  # do not know it yet
            # reconstruct templateNamespace and moduleNamespace from the first title
            if ns in templateKeys:
                colon = title.find(':')
                if colon > 1:
                    if ns == '10':
                        options.templateNamespace = title[:colon]
                        options.templatePrefix = title[:colon + 1]
                    elif ns == '828':
                        options.moduleNamespace = title[:colon]
                        options.modulePrefix = title[:colon + 1]
        if ns in templateKeys:
            text = ''.join(page)
            define_template(title, text)
            # save templates and modules to file
            if output_file:
                output.write('<page>\n')
                output.write('   <title>%s</title>\n' % title)
                output.write('   <ns>%s</ns>\n' % ns)
                output.write('   <id>%s</id>\n' % id)
                output.write('   <text>')
                for line in page:
                    output.write(line)
                output.write('   </text>\n')
                output.write('</page>\n')
        if page_count and page_count % 100000 == 0:
            logging.info("Preprocessed %d pages", page_count)
    if output_file:
        output.close()
        logging.info("Saved %d templates to '%s'", len(options.templates), output_file)


def pages_from(input):
    """
    Scans input extracting pages.
    :return: (id, revid, title, namespace key, page), page is a list of lines.
    """
    # we collect individual lines, since str.join() is significantly faster
    # than concatenation
    page = []
    id = None
    ns = '0'
    last_id = None
    revid = None
    inText = False
    redirect = False
    title = None
    for line in input:
        if not isinstance(line, text_type): line = line.decode('utf-8')
        if '<' not in line:  # faster than doing re.search()
            if inText:
                page.append(line)
                # extract categories
                if line.lstrip().startswith('[[Category:'):
                    mCat = catRE.search(line)
                    if mCat:
                        catSet.add(mCat.group(1))
            continue
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'page':
            page = []
            catSet = set()
            redirect = False
        elif tag == 'id' and not id:
            id = m.group(3)
        elif tag == 'id' and id:
            revid = m.group(3)
        elif tag == 'title':
            title = m.group(3)
        elif tag == 'ns':
            ns = m.group(3)
        elif tag == 'redirect':
            redirect = True
        elif tag == 'text':
            if m.lastindex == 3 and line[m.start(3)-2] == '/': # self closing
                # <text xml:space="preserve" />
                continue
            inText = True
            line = line[m.start(3):m.end(3)]
            page.append(line)
            if m.lastindex == 4:  # open-close
                inText = False
        elif tag == '/text':
            if m.group(1):
                page.append(m.group(1))
            inText = False
        elif inText:
            page.append(line)
        elif tag == '/page':
            if id != last_id and not redirect:
                yield (id, revid, title, ns,catSet, page)
                last_id = id
                ns = '0'
            id = None
            revid = None
            title = None
            page = []


def process_dump(input_file, template_file, out_file, file_size, file_compress,
                 process_count):
    """
    :param input_file: name of the wikipedia dump file; '-' to read from stdin
    :param template_file: optional file with template definitions.
    :param out_file: directory where to store extracted data, or '-' for stdout
    :param file_size: max size of each extracted file, or None for no max (one file)
    :param file_compress: whether to compress files with bzip.
    :param process_count: number of extraction processes to spawn.
    """

    if input_file == '-':
        input = sys.stdin
    else:
        input = fileinput.FileInput(input_file, openhook=fileinput.hook_compressed)

    # collect siteinfo
    for line in input:
        # When an input file is .bz2 or .gz, line can be a bytes even in Python 3.
        if not isinstance(line, text_type): line = line.decode('utf-8')
        m = tagRE.search(line)
        if not m:
            continue
        tag = m.group(2)
        if tag == 'base':
            # discover urlbase from the xml dump file
            # /mediawiki/siteinfo/base
            base = m.group(3)
            options.urlbase = base[:base.rfind("/")]
        elif tag == 'namespace':
            mk = keyRE.search(line)
            if mk:
                nsid = ''.join(mk.groups())
            else:
                nsid = ''
            options.knownNamespaces[m.group(3)] = nsid
            if re.search('key="10"', line):
                options.templateNamespace = m.group(3)
                options.templatePrefix = options.templateNamespace + ':'
            elif re.search('key="828"', line):
                options.moduleNamespace = m.group(3)
                options.modulePrefix = options.moduleNamespace + ':'
        elif tag == '/siteinfo':
            break

    if options.expand_templates:
        # preprocess
        template_load_start = default_timer()
        if template_file:
            if os.path.exists(template_file):
                logging.info("Loading template definitions from: %s", template_file)
                # can't use with here:
                file = fileinput.FileInput(template_file,
                                           openhook=fileinput.hook_compressed)
                load_templates(file)
                file.close()
            else:
                if input_file == '-':
                    # can't scan then reset stdin; must error w/ suggestion to specify template_file
                    raise ValueError("to use templates with stdin dump, must supply explicit template-file")
                logging.info("Preprocessing '%s' to collect template definitions: this may take some time.", input_file)
                load_templates(input, template_file)
                input.close()
                input = fileinput.FileInput(input_file, openhook=fileinput.hook_compressed)
        template_load_elapsed = default_timer() - template_load_start
        logging.info("Loaded %d templates in %.1fs", len(options.templates), template_load_elapsed)

    # process pages
    logging.info("Starting page extraction from %s.", input_file)
    extract_start = default_timer()

    # Parallel Map/Reduce:
    # - pages to be processed are dispatched to workers
    # - a reduce process collects the results, sort them and print them.

    process_count = max(1, process_count)
    maxsize = 10 * process_count
    # output queue
    output_queue = Queue(maxsize=maxsize)

    if out_file == '-':
        out_file = None

    worker_count = process_count

    # load balancing
    max_spool_length = 10000
    spool_length = Value('i', 0, lock=False)

    # reduce job that sorts and prints output
    reduce = Process(target=reduce_process,
                     args=(options, output_queue, spool_length,
                           out_file, file_size, file_compress))
    reduce.start()

    # initialize jobs queue
    jobs_queue = Queue(maxsize=maxsize)

    # start worker processes
    logging.info("Using %d extract processes.", worker_count)
    workers = []
    for i in range(worker_count):
        extractor = Process(target=extract_process,
                            args=(options, i, jobs_queue, output_queue))
        extractor.daemon = True  # only live while parent process lives
        extractor.start()
        workers.append(extractor)

    # Mapper process
    page_num = 0
    for page_data in pages_from(input):
        id, revid, title, ns, catSet, page = page_data
        if keepPage(ns, catSet, page):
            # slow down
            delay = 0
            if spool_length.value > max_spool_length:
                # reduce to 10%
                while spool_length.value > max_spool_length/10:
                    time.sleep(10)
                    delay += 10
            if delay:
                logging.info('Delay %ds', delay)
            job = (id, revid, title, page, page_num)
            jobs_queue.put(job) # goes to any available extract_process
            page_num += 1
        page = None             # free memory

    input.close()

    # signal termination
    for _ in workers:
        jobs_queue.put(None)
    # wait for workers to terminate
    for w in workers:
        w.join()

    # signal end of work to reduce process
    output_queue.put(None)
    # wait for it to finish
    reduce.join()

    extract_duration = default_timer() - extract_start
    extract_rate = page_num / extract_duration
    logging.info("Finished %d-process extraction of %d articles in %.1fs (%.1f art/s)",
                 process_count, page_num, extract_duration, extract_rate)
    logging.info("total of page: %d, total of articl page: %d; total of used articl page: %d" % (g_page_total, g_page_articl_total,g_page_articl_used_total))


# ----------------------------------------------------------------------
# Multiprocess support


def extract_process(opts, i, jobs_queue, output_queue):
    """Pull tuples of raw page content, do CPU/regex-heavy fixup, push finished text
    :param i: process id.
    :param jobs_queue: where to get jobs.
    :param output_queue: where to queue extracted text for output.
    """

    global options
    options = opts

    createLogger(options.quiet, options.debug, options.log_file)

    out = StringIO()                 # memory buffer


    while True:
        job = jobs_queue.get()  # job is (id, title, page, page_num)
        if job:
            id, revid, title, page, page_num = job
            try:
                e = Extractor(*job[:4]) # (id, revid, title, page)
                page = None              # free memory
                e.extract(out)
                text = out.getvalue()
            except:
                text = ''
                logging.exception('Processing page: %s %s', id, title)

            output_queue.put((page_num, text))
            out.truncate(0)
            out.seek(0)
        else:
            logging.debug('Quit extractor')
            break
    out.close()


report_period = 10000           # progress report period
def reduce_process(opts, output_queue, spool_length,
                   out_file=None, file_size=0, file_compress=True):
    """Pull finished article text, write series of files (or stdout)
    :param opts: global parameters.
    :param output_queue: text to be output.
    :param spool_length: spool length.
    :param out_file: filename where to print.
    :param file_size: max file size.
    :param file_compress: whether to compress output.
    """

    global options
    options = opts

    createLogger(options.quiet, options.debug, options.log_file)

    if out_file:
        nextFile = NextFile(out_file)
        output = OutputSplitter(nextFile, file_size, file_compress)
    else:
        output = sys.stdout if PY2 else sys.stdout.buffer
        if file_compress:
            logging.warn("writing to stdout, so no output compression (use an external tool)")

    interval_start = default_timer()
    # FIXME: use a heap
    spool = {}        # collected pages
    next_page = 0     # sequence numbering of page
    while True:
        if next_page in spool:
            output.write(spool.pop(next_page).encode('utf-8'))
            next_page += 1
            # tell mapper our load:
            spool_length.value = len(spool)
            # progress report
            if next_page % report_period == 0:
                interval_rate = report_period / (default_timer() - interval_start)
                logging.info("Extracted %d articles (%.1f art/s)",
                             next_page, interval_rate)
                interval_start = default_timer()
        else:
            # mapper puts None to signal finish
            pair = output_queue.get()
            if not pair:
                break
            page_num, text = pair
            spool[page_num] = text
            # tell mapper our load:
            spool_length.value = len(spool)
            # FIXME: if an extractor dies, process stalls; the other processes
            # continue to produce pairs, filling up memory.
            if len(spool) > 200:
                logging.debug('Collected %d, waiting: %d, %d', len(spool),
                              next_page, next_page == page_num)
    if output != sys.stdout:
        output.close()


# ----------------------------------------------------------------------

# Minimum size of output files
minFileSize = 200 * 1024

def main():

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument("input",
                        help="XML wiki dump file")
    groupO = parser.add_argument_group('Output')
    groupO.add_argument("-o", "--output", default="text",
                        help="directory for extracted files (or '-' for dumping to stdout)")
    groupO.add_argument("-b", "--bytes", default="1M",
                        help="maximum bytes per output file (default %(default)s)",
                        metavar="n[KMG]")
    groupO.add_argument("-c", "--compress", action="store_true",
                        help="compress output files using bzip")
    groupO.add_argument("--json", action="store_true",
                        help="write output in json format instead of the default one")


    groupP = parser.add_argument_group('Processing')
    groupP.add_argument("--html", action="store_true",
                        help="produce HTML output, subsumes --links")
    groupP.add_argument("-l", "--links", action="store_true",
                        help="preserve links")
    groupP.add_argument("-s", "--sections", action="store_true",
                        help="preserve sections")
    groupP.add_argument("--lists", action="store_true",
                        help="preserve lists")
    groupP.add_argument("-ns", "--namespaces", default="", metavar="ns1,ns2",
                        help="accepted namespaces in links")
    groupP.add_argument("--templates",
                        help="use or create file containing templates")
    groupP.add_argument("--no_templates", action="store_false",
                        help="Do not expand templates")
    groupP.add_argument("-r", "--revision", action="store_true", default=options.print_revision,
                        help="Include the document revision id (default=%(default)s)")
    groupP.add_argument("--min_text_length", type=int, default=options.min_text_length,
                        help="Minimum expanded text length required to write document (default=%(default)s)")
    groupP.add_argument("--filter_disambig_pages", action="store_true", default=options.filter_disambig_pages,
                        help="Remove pages from output that contain disabmiguation markup (default=%(default)s)")
    groupP.add_argument("-it", "--ignored_tags", default="", metavar="abbr,b,big",
                        help="comma separated list of tags that will be dropped, keeping their content")
    groupP.add_argument("-de", "--discard_elements", default="", metavar="gallery,timeline,noinclude",
                        help="comma separated list of elements that will be removed from the article text")
    groupP.add_argument("--keep_tables", action="store_true", default=options.keep_tables,
                        help="Preserve tables in the output article text (default=%(default)s)")
    default_process_count = max(1, cpu_count() - 1)
    parser.add_argument("--processes", type=int, default=default_process_count,
                        help="Number of processes to use (default %(default)s)")

    groupS = parser.add_argument_group('Special')
    groupS.add_argument("-q", "--quiet", action="store_true",
                        help="suppress reporting progress info")
    groupS.add_argument("--debug", action="store_true",
                        help="print debug info")
    groupS.add_argument("-a", "--article", action="store_true",
                        help="analyze a file containing a single article (debug option)")
    groupS.add_argument("--log_file",
                        help="path to save the log info")
    groupS.add_argument("-v", "--version", action="version",
                        version='%(prog)s ' + version,
                        help="print program version")
    groupP.add_argument("--filter_category",
                        help="specify the file that listing the Categories you want to include or exclude. One line for"
                             " one category. starting with: 1) '#' comment, ignored; 2) '^' exclude; Note: excluding has higher priority than including")
    args = parser.parse_args()

    options.keepLinks = args.links
    options.keepSections = args.sections
    options.keepLists = args.lists
    options.toHTML = args.html
    options.write_json = args.json
    options.print_revision = args.revision
    options.min_text_length = args.min_text_length
    if args.html:
        options.keepLinks = True

    options.expand_templates = args.no_templates
    options.filter_disambig_pages = args.filter_disambig_pages
    options.keep_tables = args.keep_tables

    try:
        power = 'kmg'.find(args.bytes[-1].lower()) + 1
        file_size = int(args.bytes[:-1]) * 1024 ** power
        if file_size < minFileSize:
            raise ValueError()
    except ValueError:
        logging.error('Insufficient or invalid size: %s', args.bytes)
        return

    if args.namespaces:
        options.acceptedNamespaces = set(args.namespaces.split(','))

    # ignoredTags and discardElemets have default values already supplied, if passed in the defaults are overwritten
    if args.ignored_tags:
        ignoredTags = set(args.ignored_tags.split(','))
    else:
        ignoredTags = [
            'abbr', 'b', 'big', 'blockquote', 'center', 'cite', 'em',
            'font', 'h1', 'h2', 'h3', 'h4', 'hiero', 'i', 'kbd',
            'p', 'plaintext', 's', 'span', 'strike', 'strong',
            'tt', 'u', 'var'
        ]

    # 'a' tag is handled separately
    for tag in ignoredTags:
        ignoreTag(tag)

    if args.discard_elements:
        options.discardElements = set(args.discard_elements.split(','))

    FORMAT = '%(levelname)s: %(message)s'
    logging.basicConfig(format=FORMAT)

    options.quiet = args.quiet
    options.debug = args.debug
    options.log_file = args.log_file
    createLogger(options.quiet, options.debug, options.log_file)

    input_file = args.input

    if not options.keepLinks:
        ignoreTag('a')

    # sharing cache of parser templates is too slow:
    # manager = Manager()
    # templateCache = manager.dict()

    if args.article:
        if args.templates:
            if os.path.exists(args.templates):
                with open(args.templates) as file:
                    load_templates(file)

        file = fileinput.FileInput(input_file, openhook=fileinput.hook_compressed)
        for page_data in pages_from(file):
            id, revid, title, ns,catSet, page = page_data
            Extractor(id, revid, title, page).extract(sys.stdout)
        file.close()
        return

    output_path = args.output
    if output_path != '-' and not os.path.isdir(output_path):
        try:
            os.makedirs(output_path)
        except:
            logging.error('Could not create: %s', output_path)
            return

    filter_category = args.filter_category
    if (filter_category != None and len(filter_category)>0):
        with open(filter_category) as f:
            error_cnt = 0
            for line in f.readlines():
                try:
                    line = str(line.strip())
                    if line.startswith('#') or len(line) == 0:
                        continue;
                    elif line.startswith('^'):
                        options.filter_category_exclude.add(line.lstrip('^'))
                    else:
                        options.filter_category_include.add(line)
                except Exception as e:
                    error_cnt += 1
                    print(u"Category not in utf8, ignored. error cnt %d:\t%s" % (error_cnt,e))
                    print(line)
            logging.info("Excluding categories:",)
            logging.info(str(options.filter_category_exclude))
            logging.info("Including categories:")
            logging.info(str(len(options.filter_category_include)))

    process_dump(input_file, args.templates, output_path, file_size,
                 args.compress, args.processes)

def createLogger(quiet, debug, log_file):
    logger = logging.getLogger()
    if not quiet:
        logger.setLevel(logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)
    #print (log_file)
    if log_file:
        fileHandler = logging.FileHandler(log_file)
        logger.addHandler(fileHandler)

if __name__ == '__main__':
    main()
