import argparse
import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflect
import logging
import os
import image_retrieval
import t2v_pb2 as t2v
import t2v_pb2_grpc as t2v_grpc
import grpc_server

_SERVICE_NAME = 'ImageSearch'
_INDEX_ENV_VAR = 'INDEX_DIR'
_PORT_ENV_VAR = 'PORT'


def parse_argv():
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        'index_dir',
        nargs='?',
        default='faiss_index',
        help='source directory of the FAISS index storing the image features and urls'
    )
    main_parser.add_argument(
        'tern_model',
        nargs='?',
        default='tern_data/model_tern_teran_uncertainty_best_ndcgspice.pth.tar',
        help='TERN model checkpoint'
    )
    main_parser.add_argument(
        '--port',
        type=int,
        default=8061,
        help='Port where the server should listen (defaults to 8061)'
    )
    return main_parser.parse_args()


def run_pull_mode(ir_object, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    t2v_grpc.add_ImageSearchServicer_to_server(
        grpc_server.GrpcServer(ir_object),
        server)
    SERVICE_NAME = (
        t2v.DESCRIPTOR.services_by_name[_SERVICE_NAME].full_name,
        grpc_reflect.SERVICE_NAME
    )
    grpc_reflect.enable_server_reflection(SERVICE_NAME, server)
    server.add_insecure_port(f'[::]:{port}')
    logging.info('Starting server at [::]:%d', port)
    server.start()
    server.wait_for_termination()


def main():
    args = parse_argv()
    index_dir = args.index_dir
    tern_model = args.tern_model
    index_dir = os.getenv(_INDEX_ENV_VAR, index_dir)
    port = args.port
    port = os.getenv(_PORT_ENV_VAR, port)
    logging.info('Instantiating Image Search Object...')

    ir = image_retrieval.ImageRetrieval(index_dir, tern_model)

    run_pull_mode(ir, port)


if __name__ == '__main__':
    logging.basicConfig(
        format='[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)
    main()