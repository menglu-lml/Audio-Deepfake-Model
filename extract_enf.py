class extract_enf:

    def __init__(self, signal0, fs=1000, frame_size_secs=0.2, overlap_amount_secs=0, nfft=4096, nominal=None,
                 harmonic_multiples=None, duration=None, width_band=2, width_signal=0.5, strip_index=0):

        self.signal0 = signal0
        self.fs = fs  
        self.frame_size_secs = frame_size_secs  
        self.overlap_amount_secs = overlap_amount_secs
        self.nfft = nfft
        self.nominal = nominal  
        self.harmonic_multiples = np.arange(harmonic_multiples + 1)
        self.harmonic_multiples = self.harmonic_multiples[1:len(self.harmonic_multiples)] 
        self.duration = duration  
        self.harmonics = np.multiply(self.nominal, self.harmonic_multiples)
        self.width_band = width_band  
        self.width_signal = width_signal  
        self.strip_index = strip_index  

    def find_closest(self, list_of_values, value):
        index = 1
        for i in range(1, len(list_of_values) + 1):
            if (abs(list_of_values[i] - value) < abs(list_of_values[i - 1] - value)):
                index = i
            else:
                break
        return index

    def QuadInterpFunction(self, vector, index):
        #print(vector)
        if max(vector) == 0:
            return 0
        if index == 0:
            index = 1
        elif index == (len(vector) - 1):
            index = len(vector) - 2

        alpha = 20 * math.log10(abs(vector[index - 1]))
        beta = 20 * math.log10(abs(vector[index]))
        gamma = 20 * math.log10(abs(vector[index + 1]))
        delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
        kmax = index
        k_star = kmax + delta
        return k_star

    def compute_spectrogam_strips(self):
        number_of_harmonics = len(self.harmonic_multiples)  
        spectro_strips = [] 
        frame_size = math.floor(self.frame_size_secs * self.fs)
        overlap_amount = self.overlap_amount_secs * self.fs
        shift_amount = frame_size - overlap_amount
        length_signal = len(self.signal0)

    
        number_of_frames = math.ceil((length_signal - frame_size + 1) / shift_amount)  

        rows = int(self.nfft / 2 + 1)
        starting = 0
        Pxx = np.zeros(shape=(rows, number_of_frames))  # declaring the PSD array
        win = signal.get_window('hamming', frame_size)  # creating a hamming window for each frame segment
        for frame in range(number_of_frames):
            ending = starting + frame_size
            x = self.signal0[starting:ending]
            f, t, P = signal.spectrogram(x, window=win, noverlap=self.overlap_amount_secs, nfft=self.nfft, fs=self.fs,
                                         mode='psd')
            Pxx[:, frame] = P[:, 0]
            starting = starting + shift_amount

        first_index = self.find_closest(f, self.nominal - self.width_band)
        second_index = self.find_closest(f, self.nominal + self.width_band)
        frequency_support = np.zeros(shape=(number_of_harmonics, 2))

        for i in range(number_of_harmonics):
            starting = first_index * self.harmonic_multiples[i]

            ending = second_index * self.harmonic_multiples[i]
            spectro_strips.append(Pxx[starting:(ending + 1), :])

            frequency_support[i, 0] = f[starting]
            frequency_support[i, 1] = f[ending]

        return spectro_strips, frequency_support

    def compute_combining_weights_from_harmonics(self):

        number_of_duration = math.ceil(len(self.signal0) / (self.duration * 60 * self.fs))
        frame_size = math.floor(self.frame_size_secs * self.fs)
        overlap_amount = self.overlap_amount_secs * self.fs
        shift_amount = frame_size - overlap_amount
        number_of_harmonics = len(self.harmonic_multiples)
        starting_frequency = self.nominal - self.width_band
        center_frequency = self.nominal
        initial_first_value = self.nominal - self.width_signal
        initial_second_value = self.nominal + self.width_signal
        weights = np.zeros(shape=(number_of_harmonics, number_of_duration))

        inside_mean = np.zeros(shape=(number_of_harmonics, number_of_duration))
        outside_mean = np.zeros(shape=(number_of_harmonics, number_of_duration))
        total_nb_frames = 0
        All_strips_Cell = []

        for dur in range(number_of_duration - 1):
            x = self.signal0[int(dur * self.duration * 60 * self.fs): int(
                min(len(self.signal0), ((dur + 1) * self.duration * 60 * self.fs + overlap_amount)))]

            number_of_frames = math.ceil((len(x) - frame_size + 1) / shift_amount) 

            rows = int(self.nfft / 2 + 1)
            starting = 0
            Pxx = np.zeros(shape=(rows, number_of_frames))  
            win = signal.get_window('hamming', frame_size)  

            for frame in range(number_of_frames):
                ending = starting + frame_size
                sig = self.signal0[starting:ending]
                f, t, P = signal.spectrogram(sig, window=win, noverlap=self.overlap_amount_secs, nfft=self.nfft,
                                             fs=self.fs, mode='psd')
                Pxx[:, frame] = P[:, 0]
                starting = starting + shift_amount

            width_init = self.find_closest(f, center_frequency) - self.find_closest(f, starting_frequency)
            HarmonicStrips = np.zeros(shape=((width_init * 2 * sum(self.harmonic_multiples)), number_of_frames))
            FreqAxis = np.zeros(shape=((width_init * 2 * sum(self.harmonic_multiples)), 1))
            resolution = f[1] - f[0]

            starting = 0
            starting_indices = np.zeros(shape=(number_of_harmonics, 1))
            ending_indices = np.zeros(shape=(number_of_harmonics, 1))

            for k in range(number_of_harmonics):
                starting_indices[k] = starting
                width = width_init * self.harmonic_multiples[k]
                ending = starting + 2 * width

                ending_indices[k] = ending

                tempFreqIndex = round(self.harmonics[k] / resolution)

                st = int(tempFreqIndex - width)
                en = int(tempFreqIndex + width)

                HarmonicStrips[starting:ending, :] = Pxx[st:en, :]
                FreqAxis[starting:ending, 0] = f[st:en]
                starting = ending

            All_strips_Cell.append(HarmonicStrips)


            for k in range(number_of_harmonics):
                currStrip = HarmonicStrips[int(starting_indices[k]):int(ending_indices[k]), :]
                freq_axis = FreqAxis[int(starting_indices[k]):int(ending_indices[k])]

                first_value = initial_first_value * self.harmonic_multiples[k]
                second_value = initial_second_value * self.harmonic_multiples[k]

                first_index = self.find_closest(freq_axis, first_value)
                second_index = self.find_closest(freq_axis, second_value)

                second_index = second_index + 1

                inside_strip = currStrip[first_index:second_index, :]
                inside_mean[k, dur] = np.mean(inside_strip)


                outside_strip1 = currStrip[0:first_index, :]
                outside_strip2 = currStrip[second_index:len(currStrip), :]
                outside_mean[k, dur] = np.mean(np.append(outside_strip1, outside_strip2))
                if inside_mean[k, dur] < outside_mean[k, dur]:
                    weights[k, dur] = 0
                else:
                    weights[k, dur] = inside_mean[k, dur] / outside_mean[k, dur]

            sum_weights = np.sum(weights[:, dur], axis=0)

            for k in range(number_of_harmonics):
                weights[k, dur] = (100 * weights[k, dur]) / sum_weights

        return weights

    def compute_combined_spectrum(self, strips, weights, freq_support):

        number_of_duration = (np.shape(weights))[1]
        number_of_frames = (np.shape(strips[0]))[1]
        number_of_frames_per_duration = (self.duration * 60) / self.frame_size_secs
        strip_width = np.shape((strips[self.strip_index]))[0]
        OurStripCell = []
        number_of_signals = np.shape(strips)[0]
        initial_frequency = freq_support[0, 0]

        begin = 0
        for dur in range(number_of_duration):
            number_of_frames_left = number_of_frames - dur * number_of_frames_per_duration
            OurStrip = np.zeros(
                shape=(int(strip_width), min(int(number_of_frames_per_duration), int(number_of_frames_left))))
            endit = begin + (np.shape(OurStrip))[1]
            for harm in range(number_of_signals):
                tempStrip = (strips[harm])[:, begin:endit]
                q = (np.shape(OurStrip))[1]
                for frame in range(q):
                    temp = tempStrip[:, frame:(frame + 1)]
                    tempo = cv2.resize(temp, (1,strip_width), interpolation=cv2.INTER_LINEAR)
                    tempo = 100 * tempo / max(tempo)
                    OurStrip[:, frame:(frame + 1)] = OurStrip[:, frame:(frame + 1)] + (weights[harm, dur] * tempo)

            OurStripCell.append(OurStrip)
            begin = endit

        return OurStripCell, initial_frequency



    def compute_ENF_from_combined_strip(self, OurStripCell, initial_frequency):

        number_of_duration = len(OurStripCell)
        number_of_frames_per_dur = ((OurStripCell[0]).shape)[1]
        number_of_frames = number_of_frames_per_dur * (number_of_duration - 1) + ((OurStripCell[0]).shape)[1]
        ENF = np.zeros(shape=(number_of_frames, 1))

        starting = 0
        for dur in range(number_of_duration):
            OurStrip_here = OurStripCell[dur]
            number_of_frames_here = (OurStrip_here.shape)[1]
            ending = starting + number_of_frames_here
            ENF_here = np.zeros(shape=(number_of_frames_here, 1))
            for frame in range(number_of_frames_here):
                power_vector = OurStrip_here[:, frame]
                list_power_vector = list(power_vector)
                index = list_power_vector.index(max(list_power_vector))
                k_star = self.QuadInterpFunction(power_vector, index)
                ENF_here[frame] = initial_frequency + self.fs*(k_star/self.nfft)
            ENF[starting:ending] = ENF_here
            starting = ending
        return ENF