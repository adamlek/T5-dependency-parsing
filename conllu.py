from __future__ import print_function, unicode_literals
from __future__ import unicode_literals

import re
from collections import OrderedDict, defaultdict

try:
    from io import StringIO
except ImportError:
    from StringIO import StringIO

try:
    from contextlib import redirect_stdout
except ImportError:
    import sys
    import contextlib

    @contextlib.contextmanager
    def redirect_stdout(target):
        original = sys.stdout
        sys.stdout = target
        yield
        sys.stdout = original
        
        
def parse(data, fields=None, field_parsers=None):
    return [
        TokenList(*parse_token_and_metadata(sentence, fields=fields, field_parsers=field_parsers))
        for sentence in data.split("\n\n")
        if sentence
    ]

def parse_incr(in_file, fields=None, field_parsers=None):
    for sentence in _iter_sents(in_file):
        yield TokenList(*parse_token_and_metadata(sentence, fields=fields, field_parsers=field_parsers))

def parse_tree(data):
    tokenlists = parse(data)

    sentences = []
    for tokenlist in tokenlists:
        sentences.append(tokenlist.to_tree())

    return sentences

def parse_tree_incr(in_file):
    for tokenlist in parse_incr(in_file):
        yield tokenlist.to_tree()

def _iter_sents(in_file):
    buf = []
    for line in in_file:
        if line == "\n":
            yield "".join(buf)[:-1]
            buf = []
        else:
            buf.append(line)
    if buf:
        yield "".join(buf)

def capture_print(func, args=None):
    f = StringIO()
    with redirect_stdout(f):
        if args:
            func(args)
        else:
            func()

    return f.getvalue()


try:
    unicode('')
except NameError:
    unicode = str

def text(value):
    return unicode(value)

DEFAULT_EXCLUDE_FIELDS = ('id', 'deprel', 'xpostag', 'feats', 'head', 'deps', 'misc')


class TokenList(list):
    metadata = None

    def __init__(self, tokens, metadata=None):
        super(TokenList, self).__init__(tokens)
        if not isinstance(tokens, list):
            raise ParseException("Can't create TokenList, tokens is not a list.")

        self.metadata = metadata

    def __repr__(self):
        return 'TokenList<' + ', '.join(token['form'] for token in self) + '>'

    def __eq__(self, other):
        return super(TokenList, self).__eq__(other) \
            and (not hasattr(other, 'metadata') or self.metadata == other.metadata)

    def __ne__(self, other):
        return not self == other

    def clear(self):
        self[:] = []  # Supported in Python 2 and 3, unlike clear()
        self.metadata = None

    def copy(self):
        tokens_copy = self[:]  # Supported in Python 2 and 3, unlike copy()
        return TokenList(tokens_copy, self.metadata)

    def extend(self, iterable):
        super(TokenList, self).extend(iterable)
        if hasattr(iterable, 'metadata'):
            if hasattr(self.metadata, '__add__') and hasattr(iterable.metadata, '__add__'):
                self.metadata += iterable.metadata
            elif type(self.metadata) is dict and type(iterable.metadata) is dict:
                # noinspection PyUnresolvedReferences
                self.metadata.update(iterable.metadata)
            else:
                self.metadata = [self.metadata, iterable.metadata]

    @property
    def tokens(self):
        return self

    @tokens.setter
    def tokens(self, value):
        self[:] = value  # Supported in Python 2 and 3, unlike clear()

    def serialize(self):
        return serialize(self)

    def to_tree(self):
        def _create_tree(head_to_token_mapping, id_=0):
            return [
                TokenTree(child, _create_tree(head_to_token_mapping, child["id"]))
                for child in head_to_token_mapping[id_]
            ]

        root = _create_tree(head_to_token(self))[0]
        root.set_metadata(self.metadata)
        return root


class TokenTree(object):
    token = None
    children = None
    metadata = None

    def __init__(self, token, children, metadata=None):
        self.token = token
        self.children = children
        self.metadata = metadata

    def set_metadata(self, metadata):
        self.metadata = metadata

    def __repr__(self):
        return 'TokenTree<' + \
            'token={id=' + text(self.token['id']) + ', form=' + self.token['form'] + '}, ' + \
            'children=' + ('[...]' if self.children else 'None') + \
            '>'

    def __eq__(self, other):
        return self.token == other.token and self.children == other.children \
            and self.metadata == other.metadata

    def serialize(self):
        if not self.token or "id" not in self.token:
            raise ParseException("Could not serialize tree, missing 'id' field.")

        def flatten_tree(root_token, token_list=[]):
            token_list.append(root_token.token)

            for child_token in root_token.children:
                flatten_tree(child_token, token_list)

            return token_list

        tokens = flatten_tree(self)
        tokens = sorted(tokens, key=lambda t: t['id'])
        tokenlist = TokenList(tokens, self.metadata)

        return serialize(tokenlist)

    def print_tree(self, depth=0, indent=4, exclude_fields=DEFAULT_EXCLUDE_FIELDS):
        if not self.token:
            raise ParseException("Can't print, token is None.")

        if "deprel" not in self.token or "id" not in self.token:
            raise ParseException("Can't print, token is missing either the id or deprel fields.")

        relevant_data = self.token.copy()
        for key in exclude_fields:
            if key in relevant_data:
                del relevant_data[key]

        node_repr = ' '.join([
            '{key}:{value}'.format(key=key, value=value)
            for key, value in relevant_data.items()
        ])

        print(' ' * indent * depth + '(deprel:{deprel}) {node_repr} [{idx}]'.format(
            deprel=self.token['deprel'],
            node_repr=node_repr,
            idx=self.token['id'],
        ))
        for child in self.children:
            child.print_tree(depth=depth + 1, indent=indent, exclude_fields=exclude_fields)



DEFAULT_FIELDS = ('id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc')
DEFAULT_FIELD_PARSERS = {
    "id": lambda line, i: parse_id_value(line[i]),
    "xpostag": lambda line, i: parse_nullable_value(line[i]),
    "feats": lambda line, i: parse_dict_value(line[i]),
    "head": lambda line, i: parse_int_value(line[i]),
    "deps": lambda line, i: parse_paired_list_value(line[i]),
    "misc": lambda line, i: parse_dict_value(line[i]),
}

def parse_token_and_metadata(data, fields=None, field_parsers=None):
    if not data:
        raise ParseException("Can't create TokenList, no data sent to constructor.")

    fields = fields or DEFAULT_FIELDS
    field_parsers = field_parsers or DEFAULT_FIELD_PARSERS

    tokens = []
    metadata = OrderedDict()

    for line in data.split('\n'):
        line = line.strip()

        if not line:
            continue

        if line.startswith('#'):
            var_name, var_value = parse_comment_line(line)
            if var_name:
                metadata[var_name] = var_value
        else:
            tokens.append(parse_line(line, fields, field_parsers))

    return tokens, metadata

def parse_line(line, fields, field_parsers=None):
    # Be backwards compatible if people called parse_line without field_parsers before
    field_parsers = field_parsers or DEFAULT_FIELD_PARSERS

    line = re.split(r"\t| {2,}", line)

    if len(line) == 1 and " " in line[0]:
        raise ParseException("Invalid line format, line must contain either tabs or two spaces.")

    data = OrderedDict()

    for i, field in enumerate(fields):
        # Allow parsing CoNNL-U files with fewer columns
        if i >= len(line):
            break

        if field in field_parsers:
            try:
                value = field_parsers[field](line, i)
            except ParseException as e:
                raise ParseException("Failed parsing field '{}': ".format(field) + str(e))

        else:
            value = line[i]

        data[field] = value

    return data

def parse_comment_line(line):
    line = line.strip()

    if line[0] != '#':
        raise ParseException("Invalid comment format, comment must start with '#'")

    stripped = line[1:].strip()
    if '=' not in line and stripped != 'newdoc' and stripped != 'newpar':
        return None, None

    name_value = line[1:].split('=', 1)
    var_name = name_value[0].strip()
    var_value = None if len(name_value) == 1 else name_value[1].strip()

    return var_name, var_value


INTEGER = re.compile(r"^0|(\-?[1-9][0-9]*)$")

def parse_int_value(value):
    if value == '_':
        return None

    if re.match(INTEGER, value):
        return int(value)
    else:
        raise ParseException("'{}' is not a valid value for parse_int_value.".format(value))


ID_SINGLE = re.compile(r"^[1-9][0-9]*$")
ID_RANGE = re.compile(r"^[1-9][0-9]*\-[1-9][0-9]*$")
ID_DOT_ID = re.compile(r"^[0-9][0-9]*\.[1-9][0-9]*$")

def parse_id_value(value):
    if not value or value == '_':
        return None

    if re.match(ID_SINGLE, value):
        return int(value)

    elif re.match(ID_RANGE, value):
        from_, to = value.split("-")
        from_, to = int(from_), int(to)
        if to > from_:
            return (int(from_), "-", int(to))

    elif re.match(ID_DOT_ID, value):
        return (int(value.split(".")[0]), ".", int(value.split(".")[1]))

    raise ParseException("'{}' is not a valid ID.".format(value))


deps_pattern = r"\d+:[a-z][a-z_-]*(:[a-z][a-z_-]*)?"
MULTI_DEPS_PATTERN = re.compile(r"^{}(\|{})*$".format(deps_pattern, deps_pattern))

def parse_paired_list_value(value):
    if re.match(MULTI_DEPS_PATTERN, value):
        return [
            (part.split(":", 1)[1], parse_int_value(part.split(":", 1)[0]))
            for part in value.split("|")
        ]

    return parse_nullable_value(value)

def parse_dict_value(value):
    if parse_nullable_value(value) is None:
        return None

    return OrderedDict([
        (part.split("=")[0], parse_nullable_value(part.split("=")[1]) if "=" in part else "")
        for part in value.split("|") if parse_nullable_value(part.split("=")[0]) is not None
    ])

def parse_nullable_value(value):
    if not value or value == "_":
        return None

    return value

def head_to_token(sentence):
    if not sentence:
        raise ParseException("Can't parse tree, need a tokenlist as input.")

    if "head" not in sentence[0]:
        raise ParseException("Can't parse tree, missing 'head' field.")

    head_indexed = defaultdict(list)
    for token in sentence:
        # Filter out range and decimal ID:s before building tree
        if "id" in token and not isinstance(token["id"], int):
            continue

        # If HEAD is negative, treat it as child of the root node
        head = max(token["head"] or 0, 0)

        head_indexed[head].append(token)

    return head_indexed

def serialize_field(field):
    if field is None:
        return '_'

    if isinstance(field, OrderedDict):
        fields = []
        for key, value in field.items():
            if value is None:
                value = "_"

            fields.append('='.join((key, value)))

        return '|'.join(fields)

    if isinstance(field, tuple):
        return "".join([text(item) for item in field])

    if isinstance(field, list):
        if len(field[0]) != 2:
            raise ParseException("Can't serialize '{}', invalid format".format(field))
        return "|".join([text(value) + ":" + text(key) for key, value in field])

    return "{}".format(field)

def serialize(tokenlist):
    lines = []

    if tokenlist.metadata:
        for key, value in tokenlist.metadata.items():
            line = "# " + key + " = " + value
            lines.append(line)

    for token_data in tokenlist:
        line = '\t'.join(serialize_field(val) for val in token_data.values())
        lines.append(line)

    return '\n'.join(lines) + "\n\n"

class ParseException(Exception):
    pass
