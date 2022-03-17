import sys
import zmq
import struct
import array
import time

import maxlab
import maxlab.system
import maxlab.chip
import maxlab.util
import maxlab.saving

from collections import namedtuple
from gym.core import Env #openai gym env

SpikeEvent = namedtuple('SpikeEvent', 'frame channel amplitude')

class MaxwellEnv(Env):
    
    """
    Environment for interfacing with the maxwell chip.
    """
    #TODO: Check that stim electrodes are within recording electrodes


    def __init__(self, config_file, electrodes, stim_electrodes, stim_frequency = .5,
                 verbose = 2,max_time = 300, spike_thresh = 5,max_stim_time = 300, filt = True):

        self.verbose = verbose
        self.start_time = time.time()
        self.cur_time = time.time()
        self.stim_time = 0
        self.max_time = max_time
        self.max_stim_time = max_stim_time

        self.array = None
        self.stim = []
        self.stim_refs = []
        self.stim_electrode_power_on = []
        self.stim_electrode_power_off = []
        self.record_chs = []

        # Used for data
        self.frame_number = None
        self.frame_data = None
        self.events_data = None

        #TODO: SET THIS
        # Stimulation parameters
        self.receive_channel = None
        self.stim_frequency = stim_frequency #frequency in Hz

        # misc
        self.spike_struct = '8xLif'
        self.spike_size = struct.calcsize(self.spike_struct)


        #Set up communication
        self.ctx = zmq.Context.instance()

        self.subscriber = self.ctx.socket(zmq.SUB)
        self.set_subscriber_settings(filt=filt)

        self.api_socket = self.ctx.socket(zmq.REQ)
        self.set_api_socket_settings()

        if verbose >= 1:
            print('Sockets configured, reading first packet:')

            
        ######### Init maxwell: #########################
        self.init_maxwell()
        self.set_spike_threshold(thresh=spike_thresh)
        
        ######### Select electrodes, then route #########
        self.reset_array()
        self.select_recording_electrodes(config_file)#TODO pass electrodes in/get from file
        self.select_stim_electrodes(stim_electrodes)
        #self.array.route()
        
        ######### Connect stimulation chs, download
        self.set_stim_electrodes(stim_electrodes)
        self.array.download()
        
        self.power_stim_electrodes()
        maxlab.util.offset()
        
        #self.power_stim_electrodes()


        # self.array = maxlab.chip.Array('online')
        # self.array.load_config(config_filename)


        # Skip first packet, bad information chance
        self.ignore_first_packet()



    def reset(self,stim_frequency = .5, max_time = 300, max_stim_time = 300):
        '''Resets the time and adjusts variables, but keeps subscriber and all
        to be the same'''
        
        self.start_time = time.time()
        self.cur_time = time.time()
        self.stim_time = 0
        
        self.stim_frequency = stim_frequency
        self.max_time = max_time
        self.max_stim_time = max_stim_time
        
        if self.verbose >= 1:
            print('Reset environment')
        

    def time_elapsed(self):
        '''Updates cur_time and returns time since the initialization'''

        self.cur_time = time.time()
        return self.cur_time - self.start_time



    def set_subscriber_settings(self,filt=True):
        """
        Generate the subscriber sockets with the same settings as the
        original C++ program.
        """
        self.subscriber.setsockopt(zmq.RCVHWM, 0)
        self.subscriber.setsockopt(zmq.RCVBUF, 10*20000*1030)
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self.subscriber.setsockopt(zmq.RCVTIMEO, 100)
        if filt:
            self.subscriber.connect('tcp://localhost:7205')
        else:
            self.subscriber.connect('tcp://localhost:7204')

    def set_api_socket_settings(self):
        """
        Generate the API socket with the same settings as the
        original C++ program.
        """
        self.api_socket = self.ctx.socket(zmq.REQ)
        self.api_socket.setsockopt(zmq.RCVTIMEO, -1)
        self.api_socket.connect('tcp://localhost:7210')


    def ignore_first_packet(self):
        '''
        This first loop ignores any partial packets to make sure the real
        loop gets aligned to an actual frame. First it spins for as long
        as recv() fails, then it waits for the RCVMORE flag to be False to
        check that the last partial frame is over.
        '''
        more = True
        while more:
            try:
                _ = self.subscriber.recv()
            except zmq.ZMQError:
                if self.time_elapsed() >= 3:
                    raise TimeoutError("Make sure the Maxwell Server is on.")
                continue
            more = self.subscriber.getsockopt(zmq.RCVMORE)
            self.cur_time = time.time()
            

        if self.verbose >=1:
            print('Successfully ignored first packet')



    def step(self):
        '''Recieve events published since last time step() was called
        This includs spike events and raw datastream'''

        self.cur_time = time.time()
        # Sometimes the publisher will be interrupted, so don't let that
        # crash the entire program, just skip the frame.
        try:
            # The first component of each message is the frame number as
            # a long long, so unpack that.
            self.frame_number = struct.unpack('Q', self.subscriber.recv())[0]

            # We're still ignoring the frame data, but we have to get it
            # out from the data stream in order to skip it.
            if self.subscriber.getsockopt(zmq.RCVMORE):
                self.frame_data = self.subscriber.recv()

            # This is the one that stores all the spikes.
            if self.subscriber.getsockopt(zmq.RCVMORE):
                self.events_data = self.subscriber.recv()

        except Exception as e:
            print(e)
            return

        # `frame_data` is a 1027-element array containing the recorded voltage
        # at each electrode, so unpack that into a usable format.
        frame = array.array('f', self.frame_data)
        # Print shape of current data frame
        #print(f'Frame number: {self.frame_number}\tFrame shape: {len(frame)}')

        self.events = []
        if self.events_data is not None:
            # The spike structure is 8 bytes of padding, a long frame
            # number, an integer channel (the amplifier, not the
            # electrode), and a float amplitude.
            
            if len(self.events_data) % self.spike_size != 0:
                print(f'Events has {len(self.events_data)} bytes,',
                    f'not divisible by {self.spike_size}', file=sys.stderr)

            # Iterate over consecutive slices of the raw events
            # data and unpack each one into a new struct.
            for i in range(0, len(self.events_data), self.spike_size):
                ev = SpikeEvent(*struct.unpack(self.spike_struct,
                    self.events_data[i:i+self.spike_size]))
                self.events.append(ev)

                # If the event is on the channel of interest, tell the
                # server to send the stimulation pattern that gets
                # defined by the provided Python script.
                if ev.channel == self.receive_channel:
                    print(f'Event detected on {self.receive_channel}. Sending stim command...')
                    self.send_command('sequence_send close_loop')


        # Just dump them to the console to show it works.
        if len(self.events) > 0:
            #print(self.events)
            pass

        # Do stimulation

        if (time.time() - self.stim_time > 1/self.stim_frequency) and (self.time_elapsed() < self.max_stim_time):
            cmd = 'sequence_send ' + self.seq_name
            self.send_command(cmd)
            self.stim_time = time.time()

            # Debugging
            if self.verbose >=2:
                print(f'Stimulating at t={self.stim_time} with command:',cmd)

        obs = None
        done = False
        if self.time_elapsed() > self.max_time:
            done = True
            
            # Debugging
            if self.verbose >=1:
                print(f'Max time {self.max_time} reached at {self.time_elapsed()}')

        return obs, done




    def send_command(self, string):
        '''
        Send a MaxWell API command to a ZMQ socket, check that the reply is
        "OK", and if not, print the error that occurred.
        '''
        self.api_socket.send_string(string)
        reply = self.api_socket.recv_string()
        if reply != 'OK':
            print(f'An error occurred {reply}.')
            self.api_socket.send_string('get_errors')
            print(self.api_socket.recv_string())
        else:
            print('Recieved',reply)

    
    def init_maxwell(self,gain=512):
        # Normal initialization of the chip
        maxlab.util.initialize()
        maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
        maxlab.send(maxlab.chip.Amplifier().set_gain(gain))

        if self.verbose >=1:
            print('Maxwell initialized')


    def set_spike_threshold(self,thresh=5):
        maxlab.send_raw(f"stream_set_event_threshold {thresh}")


    def reset_array(self, array_name = 'stimulation'):
        '''Creates array config in server with the given name'''
        
        self.array = maxlab.chip.Array(array_name)
        self.array.reset() #delete previous array
        
    
    def select_recording_electrodes(self,filename):
        '''Sets electrodes from configuration file at global path
        '''
        self.array.load_config(filename)

        if self.verbose >=1:
            print(f'Recording electroeds initialized from: {filename}')

            
    def select_stim_electrodes(self, electrodes):
        '''Selects the stimulation electrodes, must be called before routing'''
        
        self.array.select_stimulation_electrodes(electrodes)
        
    
    def set_stim_electrodes(self,electrodes):
        '''Sets stim electrodes from array'''
        

        if len(electrodes) > 32:
            raise Exception('Too many stimulation electrodes.')

        for e in electrodes:
            self.array.connect_electrode_to_stimulation(e)
            self.stim_refs.append(self.array.query_stimulation_at_electrode(e))
            
            if self.verbose >=2:
                print(f'Connected Electrode # {e}')

        for e,s in zip(electrodes,self.stim_refs):
            if not s:
                print(f"Error: electrode: {e} cannot be stimulated")

    def power_stim_electrodes(self):
        '''
        Create two parallel lists from the stim_refs which
        are the power_off and power_on commands for the electrodes
        
        Requires: Calling set_stim_electrodes()
        '''
        self.stim_electrode_power_on = []
        self.stim_electrode_power_off = []
        for s in self.stim_refs:
            self.stim_electrode_power_on.append(maxlab.chip.StimulationUnit(s).power_up(True).connect(True).set_voltage_mode().dac_source(0))
            self.stim_electrode_power_off.append(maxlab.chip.StimulationUnit(s).power_up(False))


    def create_stim_sequence(self,seq_name, amplitude=80):
        ''' Create stimulation sequence send to server, can be called from python code'''
        
        def append_stimulation_pulse(seq, amplitude,half_period=8):
             #100 samples is 5 ms for when you are appending
            #1 sample is 50 us
            # thus we want 400us, so 8 samples per half period
            
            #bits = amplitude//maxlab.query_DAC_lsb_mV()
            #cast as int
            
            bits = amplitude

            seq.append( maxlab.chip.DAC(0, 512-bits) )
            seq.append( maxlab.system.DelaySamples(half_period) ) 
            seq.append( maxlab.chip.DAC(0, 512+bits) )
            seq.append( maxlab.system.DelaySamples(half_period) )
            seq.append( maxlab.chip.DAC(0, 512) )
            return seq

        # Delete in case defined in server
        s = maxlab.Sequence(seq_name, persistent=True)
        del(s)

        #seqs = []
        # for i, stim in enumerate(self.stim_refs):
        #     # Create sequence
        #     seqs[i] = maxlab.Sequence(f'potter_{i}', persistent=False)
        #     seqs[i].append( self.stim_electrode_power_on[i] )
        #     seqs[i].app
        #     seqs[i].append( self.stim_electrode_power_off[i] )
        self.seq_name = seq_name

        seq = maxlab.Sequence(seq_name, persistent=True)
        seq.append( self.stim_electrode_power_on[0] )
        append_stimulation_pulse(seq, amplitude)
        seq.append( self.stim_electrode_power_off[0] )









       






if __name__ == '__main__':
    
    #Instantiate
    data_path = 'data/'


    env = MaxwellEnv()
    
    s = maxlab.saving.Saving()             # Set up file and wells for recording, 
    s.open_directory(data_path)            # I don't fully understand this code, it's taken from an example
    s.set_legacy_format(True)
    s.group_delete_all()
    
    # first maxwell
    s.group_define(0, "routed")
    s.start_file(recording_file_name)
    s.start_recording(0)
    
    done = False

    while not done:
        obs,done, = env.step()

        #obs,done, = env.step(stimulation)

        #stimulation = agent.get_stim(obs)


    s.stop_recording()
    s.stop_file()
    s.group_delete_all()


