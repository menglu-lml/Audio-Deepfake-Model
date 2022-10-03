from scipy import signal
import numpy as np
import librosa
import math
import cv2
import glob
from sklearn import svm
from torchvision import transforms

from data_processing import ADDDataset
from model import Box
from ENF import extract_ENF


def evaluation(dataset, model, device, save_path, svm):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    true_y = []
    y_pred = []
    file_name = "ADD_A_00000000"
    index = 0
    for batch_x, batch_y in data_loader:
        true_y.extend(batch_y.numpy())
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x,batch_y,is_test=True)
        batch_score = (batch_out[:, 1]
                       ).data.cpu().numpy().ravel()
        _, batch_pred = batch_out.max(dim=1)

        if batch_pred == 1:
            x = batch_x.cpu().detach().numpy()
            mysignal = extract_ENF(signal0=x[0], fs=1000, nominal=50, 
                                         harmonic_multiples=1,
                                         duration=0.05, strip_index=0)
            spectro_strip, frequency_support = mysignal.compute_spectrogam_strips()
            weights = mysignal.compute_combining_weights_from_harmonics()
            OurStripCell, initial_frequency = mysignal.compute_combined_spectrum(spectro_strip, weights, frequency_support)
            ENF = mysignal.compute_ENF_from_combined_strip(spectro_strip, initial_frequency)
            if len(ENF)>=200:
                ENF = ENF[:200]
                new_ENF = [subitem for item in ENF for subitem in item]
                prediction = svm.predict([new_ENF])
                if prediction[0] == -1:
                    batch_pred = Tensor([1]).cuda()
                else:
                    batch_pred = Tensor([0]).cuda()
        num_correct += (batch_pred == batch_y).sum(dim=0).item() 
        y_pred.extend(batch_pred.cpu().detach().numpy())

    print (100 * (num_correct / num_total))
    
    return true_y, y_pred, num_total




#### train ONE-CLASS SVM using extracted ENF
ENFVec = []

for file in glob.glob(f"/home/menglu/123/Dataset/ADD2022/ADD_train_dev/real/*.wav"):
    sig, sample_rate = librosa.load(file,sr=16000)   
    mysignal = extract_ENF(signal0=sig, fs=1000, nominal=50, 
                                     harmonic_multiples=1,
                                     duration=0.05, strip_index=0)
    spectro_strip, frequency_support = mysignal.compute_spectrogam_strips()
    weights = mysignal.compute_combining_weights_from_harmonics()
    OurStripCell, initial_frequency = mysignal.compute_combined_spectrum(spectro_strip, weights, frequency_support)
    ENF = mysignal.compute_ENF_from_combined_strip(spectro_strip, initial_frequency)
    if len(ENF)>=200:
        ENF = ENF[:200]
        ENF_vec = [subitem for item in ENF for subitem in item]
        ENFVec.append(ENF_vec)

clf = svm.OneClassSVM(nu=0.7, kernel="linear", gamma='scale')
clf.fit(ENFVec)



#### evaluation
track = 'track2'
database_path = "/home/menglu/123/Dataset/ADD2022/"+track+"adp_out"
label_path = "/home/menglu/123/Dataset/ADD2022/label/"+track+"_label.txt"
transform = transforms.Compose([

    lambda x: pad(x),
    lambda x: Tensor(x)

])
is_eval = True

evl_set = ADDDataset(data_path=database_path,label_path=label_path,is_train=False, 
                      transform=transform, is_eval=is_eval, track=track)


current_model = 'Top_path_32_512_128_12_8_0.001'
model_path = '/home/menglu/123/Deepfake/built/'+current_model+'/epoch_23.pth'
eval_output = '/home/menglu/123/Deepfake/built/'+current_model+'/eval_scores.txt'

model.load_state_dict(torch.load(model_path,map_location=device))
true_y, y_pred, num_total = evaluation(evl_set, model, device, eval_output,clf)

fpr, tpr, threshold = roc_curve(true_y, y_pred, pos_label=1)
fnr = 1 - tpr
eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
eer = (eer_1 + eer_2) / 2
print(eer)