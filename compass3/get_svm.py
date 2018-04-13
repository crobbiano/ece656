import sys
import os

sys.path.append("../")
sys.path.append("../..")
from visual_tools import VisualTools
import cv2
import numpy as np
import random
import math

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler


# ffmpeg -framerate 1 -i IMG_14%02d.JPG -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output3.mp4
if __name__ == "__main__":
    tools = VisualTools()
    trainfeats = []
    trainlabels = []
    trainimages = []
    testfeats = []
    testlabels = []
    testimages = []

    useHOG = False
    szx = 28
    szy = 28
    stridex = szx // 2
    stridey = szy // 2
    cell_size = (8, 8)
    block_size = (2, 2)
    nbins = 9

    # ----------- Generate the training set ------------
    directories = []
    directories.append(os.fsencode('./files/train/0'))
    directories.append(os.fsencode('./files/train/1'))
    directories.append(os.fsencode('./files/train/2'))
    directories.append(os.fsencode('./files/train/3'))
    directories.append(os.fsencode('./files/train/4'))
    directories.append(os.fsencode('./files/train/5'))
    directories.append(os.fsencode('./files/train/6'))
    directories.append(os.fsencode('./files/train/7'))
    directories.append(os.fsencode('./files/train/8'))
    directories.append(os.fsencode('./files/train/9'))
    neglabel = 0
    for directory in directories:
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            dirname = os.fsdecode(directory)
            filefile = os.path.join(dirname, filename)
            subimage = cv2.imread(str(filefile))
            trainimages.append(subimage)
            if useHOG:
                curr_feat, feat_size = tools.getAllHog(subimage,
                                                       stridex,
                                                       stridey,
                                                       szx,
                                                       szy,
                                                       cell_size,
                                                       block_size,
                                                       nbins,
                                                       normalize=False)
                curr_feat = curr_feat.reshape((feat_size, 1))
            else:
                subimage = cv2.cvtColor(subimage, cv2.COLOR_RGB2GRAY)
                curr_feat = subimage.reshape((784, 1))

            trainfeats.append(curr_feat)
            trainlabels.append(neglabel)
        neglabel += 1
    print("Done with negative input training")
    # ----------- END Generate the training set ------------

    # ----------- Generate the testing set ------------
    directories = []
    directories.append(os.fsencode('./files/test/0'))
    directories.append(os.fsencode('./files/test/1'))
    directories.append(os.fsencode('./files/test/2'))
    directories.append(os.fsencode('./files/test/3'))
    directories.append(os.fsencode('./files/test/4'))
    directories.append(os.fsencode('./files/test/5'))
    directories.append(os.fsencode('./files/test/6'))
    directories.append(os.fsencode('./files/test/7'))
    directories.append(os.fsencode('./files/test/8'))
    directories.append(os.fsencode('./files/test/9'))
    neglabel = 0
    # neglabel -= 1
    for directory in directories:
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            dirname = os.fsdecode(directory)
            filefile = os.path.join(dirname, filename)
            subimage = cv2.imread(str(filefile))
            testimages.append(subimage)
            # Store the test image before resizing. Its only for display purposes
            if useHOG:
                curr_feat, feat_size = tools.getAllHog(subimage,
                                                       stridex,
                                                       stridey,
                                                       szx,
                                                       szy,
                                                       cell_size,
                                                       block_size,
                                                       nbins,
                                                       normalize=False)
                curr_feat = curr_feat.reshape((feat_size, 1))
            else:
                subimage = cv2.cvtColor(subimage, cv2.COLOR_RGB2GRAY)
                curr_feat = subimage.reshape((784, 1))

            testfeats.append(curr_feat)
            testlabels.append(neglabel)
        neglabel += 1
    print("Done with negative input testing")
    # ----------- END Generate the training set ------------

    # ---------------- Randomly sample ---------------
    allfeats = trainfeats + testfeats
    alllabels = trainlabels + testlabels
    allimages = trainimages + testimages
    trainsample_idx = np.random.choice(range(len(allfeats)), int(np.floor(len(allfeats)*.10)))
    testsample_idx = np.setdiff1d(range(len(allfeats)), trainsample_idx)

    trainfeats = []
    trainlabels = []
    trainimages = []
    testfeats = []
    testlabels = []
    testimages = []
    for idx in trainsample_idx:
        trainfeats.append(allfeats[idx])
        trainlabels.append(alllabels[idx])
        trainimages.append(allimages[idx])
    for idx in testsample_idx:
        testfeats.append(allfeats[idx])
        testlabels.append(alllabels[idx])
        testimages.append(allimages[idx])
    # ---------------- END Randomly sample ---------------

    # -------- Normalization of features --------
    doNorm = True and not useHOG
    if doNorm:
        # Generate R_xx^(-.5) and mu_x from the training data and normalize train and test features
        allfeats = trainfeats + testfeats
        featsarr = np.asarray(allfeats)[:, :, 0].T
        # data_cov = np.cov(featsarr)
        # print(np.linalg.matrix_rank(data_cov))
        data_mean = np.mean(featsarr, axis=1)
        data_mean = np.expand_dims(data_mean, axis=1)
        # data_cov_inv = np.linalg.inv(data_cov)
        # (eigvals, basis)
        # v, u = np.linalg.eig(data_cov_inv)
        # vsqrt = np.sqrt(v)
        # vdiag = np.diag(vsqrt)
        # data_cov_inv_sqrt = u@vdiag@u.T
        trainfeatsnorm = []
        testfeatsnorm = []
        for featidx, feat in enumerate(trainfeats):
            trainfeatsnorm.append(trainfeats[featidx] - data_mean)
            trainfeatsnorm[featidx] = trainfeatsnorm[featidx]/np.std(trainfeatsnorm[featidx])
            # trainfeatsnorm[featidx] = data_cov_inv_sqrt@trainfeatsnorm[featidx]
        for featidx, feat in enumerate(testfeats):
            testfeatsnorm.append(testfeats[featidx] - data_mean)
            testfeatsnorm[featidx] = testfeatsnorm[featidx]/np.std(testfeatsnorm[featidx])
            # testfeatsnorm[featidx] = data_cov_inv_sqrt@testfeatsnorm[featidx]

        # Check for normality
        # featsarr_check = np.asarray(trainfeatsnorm)[:, :, 0].T
        # data_cov_check = np.cov(featsarr_check)
    # -------- END Normalization of features --------


    all_labels = sorted(list(set(testlabels)), reverse=True)
    all_labels_words = ['0','1','2','3','4','5','6','7','8','9']
    # Use bag of words in the SVM


    trainfeats = np.float32(trainfeats)
    if doNorm:
        trainfeats = np.float32(trainfeatsnorm)
    trainlabels = np.array(trainlabels)
    trainlabels = trainlabels.reshape(trainlabels.shape[0], 1)

    testfeats = np.float32(testfeats)
    if doNorm:
        testfeats = np.float32(testfeatsnorm)
    testlabels = np.array(testlabels)
    testlabels = testlabels.reshape(testlabels.shape[0], 1)

    # FIXME - swap train and test sets
    # tempfeats = trainfeats
    # templabels = trainlabels
    # tempimages = trainimages
    # trainfeats = testfeats
    # trainlabels = testlabels
    # trainimages = testimages
    # testfeats = tempfeats
    # testlabels = templabels
    # testimages = tempimage


    # cvals = np.linspace(2, 3, 4, 6, 7)
    cvals = np.array([1, 2, 3, 4, 6, 7])
    # cvals = np.array([1, 5, 10, 20])
    # cvals = np.array([5])
    # gvals = np.linspace(1, 1.5, 1)
    # gvals = np.array([ .1, .2, .5, 1, 2, 5])
    gvals = np.array([.2])
    # cval=10, gval=.1

    accuracy_results = []
    print("Training SVM")
    for cval in cvals:
        for gval in gvals:
            svm = cv2.ml.SVM_create()
            # svm.setKernel(cv2.ml.SVM_LINEAR)
            # svm.setKernel(cv2.ml.SVM_RBF)
            # svm.setC(cval)
            # svm.setGamma(gval)
            svm.setKernel(cv2.ml.SVM_POLY)
            svm.setDegree(cval)
            svm.setType(cv2.ml.SVM_C_SVC)
            svm.train(trainfeats, cv2.ml.ROW_SAMPLE, trainlabels)

            trainresult = svm.predict(trainfeats)[1]
            trainresult = np.array(trainresult)
            trainmask = trainresult == trainlabels
            traincorrect = np.count_nonzero(trainmask)

            result = svm.predict(testfeats)[1]
            result = np.array(result)
            mask = (result == testlabels).astype(int)
            correct = np.count_nonzero(mask)

            testcorrect_var = correct * 100.0 / result.size
            traincorrect_var = traincorrect * 100.0 / trainresult.size
            accuracy_results.append((testcorrect_var, cval, gval, svm, result, mask))
            print('cval:', cval, ' gval:', gval, ' TESTcorrect:', testcorrect_var, ' TRAINcorrect:', traincorrect_var)

    # Find the best answer
    best_svm_idx = np.argmax([tval for (tval, aa, ss, dd, rr, mm) in accuracy_results])
    (best_accuract, best_c, best_g, best_svm, result, mask) = accuracy_results[best_svm_idx]
    print("Using C=", best_c, "; g=", best_g, "; Test accuracy=", best_accuract,";")

    showimages =True
    if showimages:
        for (idx, image) in enumerate(testimages):
            if mask[idx]:
                color = (153, 50, 250)
            else:
                color = (0, 255, 0)
            cv2.putText(image, str(result[idx][0]), (0, 27), 1, 1.8, color, 1)
            cv2.imshow('img', image)
            k = cv2.waitKey(1000) & 0xff
            if k == 27:
                break

    # generate the confusion matrix
    numlabels = len(all_labels)
    falses_mat = np.zeros((numlabels, numlabels))
    confusion_stufff = []
    for label_idx, label in enumerate(all_labels):
        truthmask = (testlabels == label).astype(int)
        # mask out all non-true versions of the label - for finding Pd
        truthmaskedresults = result[truthmask > 0]
        truthresmask = (truthmaskedresults == label).astype(int)
        # otherlabels = [x for x in all_labels if x != label]
        falses = []
        for otherlabel in all_labels:
            # mask out all true versions of the label - for finding Pfa
            falseresmask = (truthmaskedresults == otherlabel).astype(int)
            falses.append(np.sum(falseresmask)/np.sum(truthmask))

        # confusion_stuff(label, classed as self, classed as other labels)
        confusion_stufff.append((label, all_labels_words[label_idx], all_labels, all_labels_words, falses))
        falses_mat[label_idx, :] = np.array(falses)


    svm.save('svm_data2.dat')
