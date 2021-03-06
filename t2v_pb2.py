# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: t2v.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='t2v.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\tt2v.proto\"*\n\x05Query\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x13\n\x0bnum_results\x18\x02 \x01(\x05\"\'\n\x07Results\x12\x0c\n\x04urls\x18\x01 \x03(\t\x12\x0e\n\x06scores\x18\x02 \x03(\x02\x32\x34\n\x0bImageSearch\x12%\n\x11TextToImageSearch\x12\x06.Query\x1a\x08.Resultsb\x06proto3'
)




_QUERY = _descriptor.Descriptor(
  name='Query',
  full_name='Query',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='text', full_name='Query.text', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='num_results', full_name='Query.num_results', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=13,
  serialized_end=55,
)


_RESULTS = _descriptor.Descriptor(
  name='Results',
  full_name='Results',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='urls', full_name='Results.urls', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='scores', full_name='Results.scores', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=57,
  serialized_end=96,
)

DESCRIPTOR.message_types_by_name['Query'] = _QUERY
DESCRIPTOR.message_types_by_name['Results'] = _RESULTS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Query = _reflection.GeneratedProtocolMessageType('Query', (_message.Message,), {
  'DESCRIPTOR' : _QUERY,
  '__module__' : 't2v_pb2'
  # @@protoc_insertion_point(class_scope:Query)
  })
_sym_db.RegisterMessage(Query)

Results = _reflection.GeneratedProtocolMessageType('Results', (_message.Message,), {
  'DESCRIPTOR' : _RESULTS,
  '__module__' : 't2v_pb2'
  # @@protoc_insertion_point(class_scope:Results)
  })
_sym_db.RegisterMessage(Results)



_IMAGESEARCH = _descriptor.ServiceDescriptor(
  name='ImageSearch',
  full_name='ImageSearch',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=98,
  serialized_end=150,
  methods=[
  _descriptor.MethodDescriptor(
    name='TextToImageSearch',
    full_name='ImageSearch.TextToImageSearch',
    index=0,
    containing_service=None,
    input_type=_QUERY,
    output_type=_RESULTS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_IMAGESEARCH)

DESCRIPTOR.services_by_name['ImageSearch'] = _IMAGESEARCH

# @@protoc_insertion_point(module_scope)
