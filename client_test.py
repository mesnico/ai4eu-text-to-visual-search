import argparse

import t2v_pb2 as t2v
import t2v_pb2_grpc as t2v_grpc
import grpc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, help='The path to the folder where bu_features resides')
    opt = parser.parse_args()

    print('Issuing the query: "{}"'.format(opt.query))

    channel = grpc.insecure_channel('localhost:8061')
    stub = t2v_grpc.ImageSearchStub(channel)

    query = t2v.Query(text=opt.query, num_results=7)
    results = stub.TextToImageSearch(query)
    print(results)

if __name__ == '__main__':
    main()