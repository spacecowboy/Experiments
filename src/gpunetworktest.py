'''
Created on Aug 18, 2011

@author: jonask

Intended to demonstrate the scaling of a genetic population of ANNs on the GPU
compared to the CPU.
'''
import numpy as np
from kalderstam.neural.network import build_feedforward, \
    build_feedforward_committee
from kalderstam.util.decorators import benchmark
import pyopencl as cl

class CL:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        print fstr
        #create the program
        self.program = cl.Program(self.ctx, fstr).build()

    def popCorn(self):
        mf = cl.mem_flags

        #initialize client side (CPU) arrays
        self.a = np.array(range(10), dtype = np.float32)
        self.b = np.array(range(10), dtype = np.float32)

        #create OpenCL buffers
        self.a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.a)
        self.b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.b)
        self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.b.nbytes)

    def gpu_single(self, P, num_of_hidden, pop_size = 1):
        mf = cl.mem_flags

        #input layer, insert 1 in first column for bias
        self.input_layer = np.asarray(np.insert(P, 0, 1, axis = 1), dtype = np.float32)
        #hidden layer, insert 1 in first column for bias
        self.hidden_layer = np.ones((pop_size, num_of_hidden + 1), dtype = np.float32)
        #output layer
        self.output_layer = np.ones((pop_size, 1), dtype = np.float32)

        #input-hidden weights, + 1 for bias
        self.hidden_weights = np.asarray(pop_size, np.random.rand(num_of_hidden, len(P[0]) + 1), dtype = np.float32)
        #hidden-output weights, + 1 for bias
        self.output_weights = np.asarray(np.random.rand(pop_size, num_of_hidden + 1), dtype = np.float32)

        #create OpenCL buffers
        self.input_layer_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = self.input_layer)

        self.hidden_layer_buf = cl.Buffer(self.ctx, mf.READ_WRITE, hostbuf = self.hidden_layer)
        self.hidden_weights_buf = cl.Buffer(self.ctx, mf.READ_WRITE, hostbuf = self.hidden_weights)
        self.output_layer_buf = cl.Buffer(self.ctx, mf.READ_WRITE, hostbuf = self.output_layer)
        self.output_weights_buf = cl.Buffer(self.ctx, mf.READ_WRITE, hostbuf = self.output_weights)

    @benchmark
    def execute_gpu_single(self):
        self.program.netsim(self.queue, self.a.shape, None, self.input_layer_buf, self.hidden_layer_buf, self.hidden_weights_buf, self.output_layer_buf, self.output_weights_buf)
        results = np.empty_like(self.output_layer)
        cl.enqueue_read_buffer(self.queue, self.output_layer_buf, results).wait()
        print "Input", self.input_layer
        print "Output", results

    @benchmark
    def execute(self):
        self.program.part1(self.queue, self.a.shape, None, self.a_buf, self.b_buf, self.dest_buf)
        c = np.empty_like(self.a)
        cl.enqueue_read_buffer(self.queue, self.dest_buf, c).wait()
        print "a", self.a
        print "b", self.b
        print "c", c

def cpu_single(P, num_of_hidden, pop_size = 1):
    #number of input covariates
    num_of_inputs = len(P[0])

    #Tanh to keep implementation details easy in opencl
    net = build_feedforward(num_of_inputs, num_of_hidden, 1, output_function = 'tanh')

    @benchmark
    def many_sim():
        for n in xrange(pop_size):
            net.sim(P)

    many_sim()

def cpu_multi(pop_size, P, num_of_hidden):
    #number of input covariates
    num_of_inputs = len(P[0])

    #Tanh to keep implementation details easy in opencl
    com = build_feedforward_committee(8, num_of_inputs, num_of_hidden, 1, output_function = 'tanh')

    benchmark(com.sim)(P)

if __name__ == '__main__':
    rows = 1000
    num_of_vars = 5
    P = np.asarray(np.random.rand(rows, num_of_vars), dtype = np.float64)
    hiddens = 1

    print("Single network")
    cpu_single(P, hiddens)
    print("Many networks")
    cpu_single(P, hiddens, 5)

    #GPU part
    print cl.get_platforms()[0].get_devices()[0]
    cl.Context(devices = cl.get_platforms()[0].get_devices())
    example = CL()
    example.loadProgram("gpunetworktest.cl")
    example.popCorn()
    example.execute()

    #example.gpu_single(P, hiddens)
    #example.execute_gpu_single()
