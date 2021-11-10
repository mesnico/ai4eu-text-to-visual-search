import logging

import t2v_pb2 as t2v
import t2v_pb2_grpc as t2v_grpc

class GrpcServer(t2v_grpc.ImageSearchServicer):
    def __init__(self, image_retrieval_object):
        self.image_retrieval_object = image_retrieval_object

    def TextToImageSearch(self, request:t2v.Query, context):
        query = request.text
        k = request.num_results
        logging.debug('Received query: {}; k={}'.format(query, k))
        urls, scores = self.image_retrieval_object.search(query, k)
        response = t2v.Results(urls=urls, scores=scores)
        return response