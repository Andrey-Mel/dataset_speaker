import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from audiomentations import * #Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from audiomentations import SpecCompose, SpecChannelShuffle, SpecFrequencyMask
import numpy as np
import librosa
from tensorflow.keras.utils import to_categorical
from tqdm.notebook import tqdm
import tensorflow as tf




# path = ''
class Generate_Data:
    
    def __init__(self, vec=True, augm=True):
        """
            Initialization
            path_data: путь к базе
            vec: True - признаки будут в векторами, False - 2D array
            augm: True - проводить аугментацию, False - не проводить аугментацию
        """
        
        self.vec = vec
        #self.path_data = path_data
        self.augm = augm
    @staticmethod
    def augmentation():    
        """
            Функция аугментации аудио данных для подачи в нейронку. 10 аугментаций одного трека аудио.
            Описания функций напротив каждого обозначания
        """
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5), #Добавления гаусовского шума
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5), # Время растягивания сигнала без изменения высоты
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5), #Изменение высоты звука вверх или вниз без изменения темпа
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),#смещение сэмпла вперед или назад
            Reverse(p=0.5), #воспроизведение в обратном направлении, как перевором изображения
            Normalize(p=0.5), #чтобы самый высокий уровень сигнала, присутствующий в звуке, стал 0 dBFS,
                             #т.е. максимально допустимый уровень громкости, если все сэмплы должны быть в диапазоне от -1 до 1.
            LowShelfFilter(min_gain_db = -4, max_gain_db=4, p=0.5),#это фильтр, который либо усиливает (увеличивает амплитуду), либо сокращает
                                                                    #(уменьшает амплитуду) частоты ниже определенной центральной частоты. Это преобразование
                                                                    #применяет фильтр нижнего уровня на определенной центральной частоте в герцах.

            HighShelfFilter(min_gain_db = -4, max_gain_db=4, p=0.5),#Что выше только для высоких частот Это преобразование применяет высокочастотный фильтр на определенной центральной частоте в герцах.
            ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=40, p=0.5),#Процент точек, которые будут отсечены, определяется на основе равномерного распределения между
                                 #два входных параметра min_percentile_threshold и max_percentile_threshold. 
            BandStopFilter(min_rolloff=12,max_rolloff=24,zero_phase=False,p=0.5),#Apply band-stop filtering to the input audio.
                                                    #zero_phase=True - Если для этого параметра установлено значение "true", это не повлияет на фазу
                                                    # входного сигнала, но будет звучать на частоте среза на 3 дБ ниже.
                                                    # по сравнению со случаем ненулевой фазы (6 дБ против 3 дБ). Кроме того,
                                                    # это в 2 раза медленнее, чем в случае ненулевой фазы.

        ])



        #Для матричных данных
        augment2 = SpecCompose(
            [
                SpecChannelShuffle(p=0.5),
                SpecFrequencyMask(p=0.5),
            ]
        )
        return augment, augment2


    #Функция параметритизация аудио в матричном виде
    def wav2conv(self,sample, sample_rate=16000):
        """
            Преобразование аудио данных в матричный вид
            input:
                y - выходные данные из либрозы
                sr - частота
        """
        
#         stft = librosa.feature.chroma_stft(y=sample,sr=sample_rate,n_fft=2048,hop_length=512,win_length=2048) #частота цветности(по умолчанию 12 баков цветности)12,216 <-216 от duration,win_length=256
#         mfcc = librosa.feature.mfcc(y=sample,sr=sample_rate)#Мел спектральные коэффициенты ( по умоччанию 20)20,216

#         rmse = (librosa.feature.rms(y=sample)) #среднеквадратич амплитуда 1,216
#         spec_cent = (librosa.feature.spectral_centroid(y=sample,sr=sample_rate))#спектральный центроид 1,216
#         spec_bw = (librosa.feature.spectral_bandwidth(y=sample,sr=sample_rate)) #ширина полосы частот 1,216
#         rolloff = (librosa.feature.spectral_rolloff(y=sample,sr=sample_rate)) #среднее спектрального спада часттоты 1,216
#         #zcr = np.mean(librosa.feature.zero_crossing_rate(y)) #частота пересечения нуля - можно получить только общее число пересечений или среднее

#         out = []
#         out = np.concatenate([mfcc,stft,spec_cent,rolloff,rmse,spec_bw],axis=0)
#         out = np.array(out)
#         out = np.pad(np.resize(out,(36,120)),((0,0),(0,0)),mode='symmetric')
#         return out
        norm_sample = librosa.util.normalize(sample)
        stfts = tf.signal.stft(norm_sample, frame_length=1024, frame_step=256, fft_length=1024)
        spectrograms = tf.abs(stfts)
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz,num_mel_bins = 80.0, 7600.0, 80
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
            spectrograms, linear_to_mel_weight_matrix,1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
        mfcc = np.pad(np.resize(mfcc,(184,80)),((0,0),(0,0)),mode='symmetric')
        return mfcc  #184x80


    #Функция параметритизация аудио в векторном виде
    def wav2mfcc(self,sample,n_fft = 1024, sample_rate=16000): #v3
        """ 
            Преобразование данных в векторном виде
            sample - audio time series
            sample_rate - sampling rate of sample
            n_fft = frame size

            добавлена функция удаления тишины - del_silent_track
        """

        mfcc = librosa.feature.mfcc(y=sample, 
                                    n_fft=n_fft, # размер фрейма
                                    window='hann',  # оконная функция (windowing)
                                    hop_length=int(n_fft*0.5), # размер перекрытия фреймов (overlapping)
                                    sr=sample_rate, 
                                    n_mfcc=20)
        features = np.mean(mfcc, axis=1)


        zero_crossings = sum(librosa.zero_crossings(sample, pad=False))
        features = np.append(zero_crossings, features)

        spec_cent = librosa.feature.spectral_centroid(y=sample,n_fft=n_fft, hop_length=int(n_fft*0.5), window='hann', sr=sample_rate).mean()
        features = np.append(spec_cent, features)

        spec_flat = librosa.feature.spectral_flatness(y=sample,n_fft=n_fft, hop_length=int(n_fft*0.5), window='hann').mean()
        features = np.append(spec_flat, features)

        spec_bw = librosa.feature.spectral_bandwidth(y=sample,n_fft=n_fft, hop_length=int(n_fft*0.5), window='hann', sr=sample_rate).mean()
        features = np.append(spec_bw, features)

        rolloff = librosa.feature.spectral_rolloff(y=sample, n_fft=n_fft, hop_length=int(n_fft*0.5), window='hann', sr=sample_rate).mean()
        features = np.append(rolloff, features)

        return features # функция вернет массив мэл-частот и массив аудио-отрезков
    
    #выходные данные векторные или массив
    
    def vec_or_conv(self,sample):
        """
            Функция формы данных - вектор или матрица
            если vec = True - это векторное представление, false - матричное представление
            Признаки из стандартной библиотеки либроза
            В матричном представлении все признаки из либрозы вытянуты в вектор и собраны в матрицу
            input: Данные из либрозы для форматирования
            data: Изменненая форма данных
        """
        if self.vec:
            data = self.wav2mfcc(sample,n_fft = 1024, sample_rate=16000)
        else:
            data = self.wav2conv(sample,sample_rate = 16000)
        return data
    
    
    

    #Визуализация аудио данных
    @staticmethod
    def plot_spec(data:np.array, sr:int, title:str, fpath:str) -> None: #(sample,sr,'origin_spec',file_path)
        """
            Функция вывода спектрограммы. Для отображения необходимо ввести вводные данные:
            input:
                data - np.array после обработки или либрозы
                sr - sample rate
                title - type string - название графика трека
                fpath - путь к файлу, но split не на длинную конструкцию пути
        """
        label = str(fpath).split('/')[-1].split('_')[0]
        fig, ax = plt.subplots(1,2,figsize=(15,5))
        ax[0].title.set_text(f'{title} / Label: {label}')
        ax[0].specgram(data, Fs=2)
        ax[1].set_ylabel('Amplitude')
        ax[1].plot(np.linspace(0,1,len(data)),data)
        
        
    #Функция dataloader
    #@staticmethod
    def dataloader(self, path_data:str):        
        """
            ИЗМЕНИЛ - цель чтобы определить 1 разз объект Класса и задавать путь в методе класса - проверить
            
            Функция выгрузки подготовленных данных для обучения, 
            но не нормированы и не разделены на test and train
            return:
                X - предподготовленные данные для обучения
                Y - классы в OHE формате
                Y_classes - классы в числовом формате
        """
        X = []
        Y = []
        augment,augment2 = self.augmentation()
        clas = os.listdir(path_data)
        #print(len(clas))
        for i in tqdm(range(len(clas)),desc = 'Собираем датасет'):
            if clas[i] not in '.DS_Store':

                path_cl = os.path.join(path_data,clas[i])
                for s in os.listdir(path_cl):
                    file = os.path.join(path_cl,s)
                    #print(file)
                    
                    #librosa load
                    amplitude,sr = librosa.load(file,mono=True, sr = 16000, duration = 3)
                    sec = amplitude.shape[0]/sr
                    if sec < 3:
                        continue
                    #выход из функции vec_to_conv 
                    data = self.vec_or_conv(amplitude)                    
                    #print(data.shape)
                    X.append(data)
                    Y.append(to_categorical(i,len(clas)))   

                    if self.augm:
                    
                        for j in range(len(augment.transforms)):
                            augment_data = augment.transforms[j](samples = amplitude,sample_rate=16000)
                            #print(augment_data.shape)
                            
                            #выход из функции vec_or_conv
                            aug_data = self.vec_or_conv(augment_data)
                            
                            #затем добавляем в dataset
                            X.append(aug_data)
                            Y.append(to_categorical(i, len(clas)))

        X = np.array(X).astype('float16')
        Y = np.array(Y).astype('float16')
        Y_classes = np.argmax(Y, axis=1).astype('float16')
        
        #возврат
        return X, Y, Y_classes
    
    #
    def dataloader_file(self,path_data:str):
        """
            ИЗМЕНИЛ - цель чтобы определить 1 разз объект Класса и задавать путь в методе класса - работает
            
            Функция получения признаков и аугментации одного файла
            path_data - путь к файлу wav
            
            Функция получения признаков и аугментации одного файла
            path_data - путь к файлу wav
            output: Выход из функции две переменные Xdata - не аугментированные данные одного файла
                    X - список аугментированых файлов, одного файла 
        
        """
        X = []
        Xdata = 0
        augment,augment2 = self.augmentation()
        amplitude,sr = librosa.load(path_data, mono=True, sr = 16000,duration = 3)
        #выход из функции vec_to_conv 
        data = self.vec_or_conv(amplitude)                    
        #print(data.shape)
        #X.append(data)
        Xdata = data
        if self.augm:

            for j in range(len(augment.transforms)):
                augment_data = augment.transforms[j](samples = amplitude,sample_rate=16000)
                #print(augment_data.shape)

                #выход из функции vec_or_conv
                aug_data = self.vec_or_conv(augment_data)

                #затем добавляем в dataset
                X.append(aug_data)
#                 X = aug_data
                

        X = np.array(X).astype('float16')
        
        #возврат
        return Xdata, X
        
