import logging

import t2v_pb2 as t2v
import t2v_pb2_grpc as t2v_grpc
import grpc

def main():
    channel = grpc.insecure_channel('localhost:8061')
    stub = t2v_grpc.ImageSearchStub(channel)

    query = t2v.Query(text='An old picture', num_results=7)
    results = stub.TextToImageSearch(query)
    print(results)

if __name__ == '__main__':
    main()