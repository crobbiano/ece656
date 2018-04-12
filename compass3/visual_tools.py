'''
visual_tools.py - Tools for the visual detection and calssification of things
'''
import numpy as np
import pandas as pd
import random
import math
import time
import cv2


class VisualTools:
    ##
    # @brief Does the initialization thing
    # @param svm The path to the SVM data to be loaded
    def __init__(self, svm=None, num_pigs_len=10):
        self.imgNum = 1
        self.num_pigs_len = num_pigs_len

        self.num_pigs = np.zeros((self.num_pigs_len, 1))

        if svm is not None:
            self.svm = cv2.ml.SVM_load(svm)
        else:
            self.svm = svm

    def isopen(self, is_pigs, percent=.99):
        self.num_pigs = np.roll(self.num_pigs, 1)
        if len(is_pigs):
            if 1 in is_pigs:
                self.num_pigs[0] = 1
            else:
                self.num_pigs[0] = 0
        else:
            self.num_pigs[0] = 0

        isopen = np.sum(self.num_pigs) / self.num_pigs_len > percent
        return isopen

    ##
    # @brief Returns the detector decision
    # @param img_og The image to get classification for
    # @param A The A matrix in the matched subspace classifier
    # @param stridex The amount of pixels to move the ROI by in x dir
    # @param stridey The amount of pixels to move the ROI by in y dir
    # @param szx The width of the image
    # @param szy The height of the image
    # @param cell_size Tuple containing the cell size, eg (8,8)
    # @param block_size Tuple containing the block size, eg (2,2).  Each block contains one cell
    # @param nbins The number of bins for the histogram
    # @return is_pig Is the image a pig or not
    # @return pig_prob Probability of being a pig
    def class_by_hog(self, img, A, stridex=32, stridey=64, szx=64, szy=128, cell_size=(8, 8), block_size=(2, 2),
                     nbins=9):
        Classed = []
        Tbinary = []
        Tthresh = 6100626580507770#2000000
        smallimg = cv2.resize(img, (128,64))
        zs = self.getAllHog(smallimg,
                            stridex,
                            stridey,
                            szx,
                            szy,
                            cell_size,
                            block_size,
                            nbins)
        #zs=zs[0]
        if zs.__len__ == 0:
            return (False, 0)

        # Process each feature by the test statistic
        zs=zs[0]
        for z in zs:
            t = z.T.dot(A.dot(z))
            print("T: %f" % t)
            Classed.append(t)
            Tbinary.append(t > Tthresh)

        # Take majority vote for the image
        pig_prob = float(sum(Tbinary) / len(Tbinary))
        is_pig = bool(pig_prob >= .5)

        cv2.imshow('small', smallimg)
        cv2.waitKey(0)

        # Return if detected
        return (is_pig, pig_prob)

    ##
    # @brief Takes in a 64x128 ROI and returns the HOG descriptors
    # @param img the 64x128 ROI
    # @param szx The width of the image
    # @param szy The height of the image
    # @param cell_size Tuple containing the cell size, eg (8,8)
    # @param block_size Tuple containing the block size, eg (2,2).  Each block contains one cell
    # @param nbins The number of bins for the histogram
    # @return status True if function was performed correctly
    # @return hog_feats Contains the HOG descriptors. Only valid is status==True
    def getHog(self, img, szx=64, szy=128, cell_size=(8, 8), block_size=(2, 2), nbins=9):
        if img.shape != (szy, szx):
            print("ERROR: getHog input image size does not match")
            return (False, -1)
        else:
            # winSize is the size of the image cropped to an multiple of the cell size
            _winSize = (img.shape[1] // cell_size[1] * cell_size[1], img.shape[0] // cell_size[0] * cell_size[0])
            _blockSize = (block_size[1] * cell_size[1], block_size[0] * cell_size[0])
            _blockStride = (cell_size[1], cell_size[0])
            _cellSize = (cell_size[1], cell_size[0])
            _nbins = nbins
            hog = cv2.HOGDescriptor(_winSize, _blockSize, _blockStride, _cellSize, _nbins)

            # hog_feats = hog.compute(img)

            n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
            hog_feats = hog.compute(img)

            return (True, hog_feats)

    ##
    # @brief Returns all HOG descriptors for an arbitrary image
    # @param img The image to get HOG descriptors for
    # @param stridex The amount of pixels to move the ROI by in x dir
    # @param stridey The amount of pixels to move the ROI by in y dir
    # @param szx The width of the image
    # @param szy The height of the image
    # @param cell_size Tuple containing the cell size, eg (8,8)
    # @param block_size Tuple containing the block size, eg (2,2).  Each block contains one cell
    # @param nbins The number of bins for the histogram
    # @param normalize Histogram normalize the grayscale image
    # @return img_feats List of HOG descriptors for all ROI within img
    def getAllHog(self, img_og, stridex=32, stridey=64, szx=64, szy=128, cell_size=(8, 8), block_size=(2, 2), nbins=9, normalize=False):
        img = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY)
        if normalize:
            clahe = cv2.createCLAHE()
            img = clahe.apply(img)
            # img = cv2.equalizeHist(img)
        w = img.shape[1]
        h = img.shape[0]

        img_feats = []
        for x in range(0, w + 1 - szx, stridex):
            for y in range(0, h + 1 - szy, stridey):
                subimg = img[y:y + szy, x:x + szx]
                # cv2.imshow('subimg',subimg)
                # cv2.waitKey(0)

                mkImages = 0
                if mkImages:
                    cv2.imwrite("output_subimages/test%000006d.png" % self.imgNum, cv2.resize(subimg, (64, 128)))
                    self.imgNum += 1

                val, hog_feats = self.getHog(subimg, szx, szy, cell_size, block_size, nbins)

                if val is True: img_feats.append(hog_feats)

        return np.array(img_feats), hog_feats.size

    ##
    # @brief Returns the SVM decision
    # @param img_og The image to get classification for
    # @param stridex The amount of pixels to move the ROI by in x dir
    # @param stridey The amount of pixels to move the ROI by in y dir
    # @param szx The width of the image
    # @param szy The height of the image
    # @param cell_size Tuple containing the cell size, eg (8,8)
    # @param block_size Tuple containing the block size, eg (2,2).  Each block contains one cell
    # @param nbins The number of bins for the histogram
    # @return class_num Is the image a pig or not
    # @return valid Is the prediction valid
    def class_by_svm(self, img_og, stridex=32, stridey=64, szx=64, szy=128, cell_size=(8, 8), block_size=(2, 2),
                     nbins=9):
        zs, feat_size = self.getAllHog(img_og, stridex, stridey, szx, szy, cell_size, block_size, nbins)

        zs = np.float32(zs)
        if self.svm is not None:
            class_num = self.svm.predict(zs)[1]
            valid = True
        else:
            print('ERROR: SVM does not exist in the tools. Answer is not valid')
            class_num = -999
            valid = False

        return (class_num, valid)

    ##
    # @brief Returns the gradients for a single image
    # @param hog_feats The HOG descriptors for the image
    # @param stridex The amount of pixels to move the ROI by in x dir
    # @param stridey The amount of pixels to move the ROI by in y dir
    # @param szx The width of the image
    # @param szy The height of the image
    # @param cell_size Tuple containing the cell size, eg (8,8)
    # @param block_size Tuple containing the block size, eg (2,2).  Each block contains one cell
    # @param nbins The number of bins for the histogram
    # @return gradients Matrix of gradients for the img
    def getGradients(self, hog_feats, stridex=32, stridey=64, szx=64, szy=128, cell_size=(8, 8), block_size=(2, 2),
                     nbins=9):
        n_cells = (szy // cell_size[0], szx // cell_size[1])
        gradients = np.zeros((n_cells[0], n_cells[1], nbins))

        # count cells (border cells appear less often across overlapping groups)
        cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

        for off_y in range(block_size[0]):
            for off_x in range(block_size[1]):
                gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                    hog_feats[:, :, off_y, off_x, :]
                cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

        # Average gradients
        gradients /= cell_count

        return gradients

    ##
    # @brief Returns all the gradients for an arbitrary image
    # @param img_feats The HOG descriptors for the image
    # @param stridex The amount of pixels to move the ROI by in x dir
    # @param stridey The amount of pixels to move the ROI by in y dir
    # @param szx The width of the image
    # @param szy The height of the image
    # @param cell_size Tuple containing the cell size, eg (8,8)
    # @param block_size Tuple containing the block size, eg (2,2).  Each block contains one cell
    # @param nbins The number of bins for the histogram
    # @return gradients List of all gradients for the img
    def getAllGradients(self, img_feats, stridex=32, stridey=64, szx=64, szy=128, cell_size=(8, 8), block_size=(2, 2),
                        nbins=9):
        gradients = []
        for feat in img_feats:
            gradients.append(self.getGradients(feat,
                                               stridex,
                                               stridey,
                                               szx,
                                               szy,
                                               cell_size,
                                               block_size,
                                               nbins,
                                               ))

        return gradients

    ##
    # @brief Calculates the pointwise distance between vectors
    # @param X A vector
    # @param Y A vector
    # @return distmat The matrix describing the pointwise distance between vectors X and Y
    def distmat(self, X, Y=None):
        if Y is None:
            Y = X

        return np.abs(X[:, None] - Y[None, :])

    ##
    # @brief Draw the flow on image
    # @param img The image to draw flow on
    # @param flow The flow for the image
    # @param step The number of pixels to skip between drawing flow
    # @param bounding_box The bounding box of the ROI in (x,y,w,h) format
    # @return vis The image with the flow drawn
    def draw_flow(self, img, flow, step=16, bounding_box=None, background=False):
        if bounding_box is None:
            h, w = img.shape[:2]
            x_offset,y_offset=0,0

        else:
            w, h = bounding_box[2], bounding_box[3]
            x_offset, y_offset = bounding_box[0], bounding_box[1]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        first = np.where(flow.any(axis=1))[0][0]
        second = np.where(flow[first].any(axis=1))[0]
        thing3 = flow[first, second]
        fx, fy = flow[y, x].T
        fx2, fy2 = flow[first, second].T
        lines = np.vstack([x + x_offset, y + y_offset, x + x_offset + fx, y + y_offset + fy]).T.reshape(-1, 2, 2)
        lines2 = np.vstack([second + x_offset, first + y_offset, second + x_offset + fx2, first + y_offset + fy2]).T.reshape(-1, 2, 2)
        # lines = np.int32(lines + 0.5)
        lines = np.int32(lines)
        lines2 = np.int32(lines2)
        vis = img.copy()
        # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines2, 0, (0, 255, 255))
        # cv2.polylines(vis, lines, 0, (0, 255, 255))
        if background is not False:
            for (x1, y1), (_x2, _y2) in lines:
                cv2.circle(vis, (x1, y1), 7, (153, 50, 250), -1)
        return vis

    ##
    # @brief Find connected regions in a binary mask
    # @param detected A binary image
    # @param flow The OF for the image
    # @return mask The total mask of the flow
    # @return inv_mask The inverse of mask
    # @return num_labels The number of regions
    # @return labels The matrix containing the regions
    # @return stats The stats for each region - noted in the code comments
    def findregions(self, detected, flow=None):
        # change detected matrix to uint8 for use with connectedComponents
        im = np.array(detected * 255, dtype=np.uint8)
        ret, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow('det', im)

        connectivity = 8
        output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

        num_labels = output[0]
        labels = output[1]
        stats = output[2]
        centroids = output[3]
        # Statistics output for each label, including the background label, see below for available statistics.
        # Statistics are accessed via stats[label, COLUMN] where available columns are defined below.
        #
        # cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
        # cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
        # cv2.CC_STAT_WIDTH The horizontal size of the bounding box
        # cv2.CC_STAT_HEIGHT The vertical size of the bounding box
        # cv2.CC_STAT_AREA The total area (in pixels) of the connected component

        if flow is not None:
            # Use the bounding box from each connected section to extract the flow for that box
            # and then calculate the average direction for the entire box
            # Logical AND the detected binary map with the flow to get flow_mask which contains the thresholded flow
            # flow_mask = np.dstack([detected, detected])
            indiv_flow = []
            avg_indiv_flow = []
            for idx in range(0, num_labels):
                x = stats[idx, 0]
                y = stats[idx, 1]
                w = stats[idx, 2]
                h = stats[idx, 3]

                # indiv_flow.append(flow_mask[y:y+h, x:x+w])

                local_flow = flow[y:y+h, x:x+w]
                mag, ang = cv2.cartToPolar(local_flow[..., 0], local_flow[..., 1])
                # mag_avg = np.mean(mag)
                mag_avg = float(mag.max())
                mag_ave_idx = np.argmax(mag)
                # ang_avg = np.mean(ang)
                ang_avg = float(ang.item(mag_ave_idx))
                avg_flow = cv2.polarToCart(np.array([mag_avg]), np.array([ang_avg]))

                x1,y1 = (int(centroids[idx][0]), int(centroids[idx][1]))
                # avg_indiv_flow.append((x1, y1, avg_flow[0][0][0], avg_flow[1][0][0]))
                # avg_indiv_flow.append(avg_flow)
                yset = y1-y
                xset = x1-x
                real_local = np.zeros_like(local_flow)
                real_local[y1-y, x1-x, 0] = avg_flow[0][0][0]*10
                real_local[y1-y, x1-x, 1] = avg_flow[1][0][0]*10
                indiv_flow.append(local_flow)
                avg_indiv_flow.append(real_local)
                # indiv_flow.append(local_flow)

            # avg_indiv_flow = np.array([avg_indiv_flow])
            # lines = np.int32(avg_indiv_flow + 0.5)

        mask = (labels >= 1).astype(int)
        inv_mask = (labels == 0).astype(int)
        mask = np.array(mask, dtype=np.uint8)
        inv_mask = np.array(inv_mask, dtype=np.uint8)

        if flow is not None:
            return (mask, inv_mask, num_labels, labels, stats, centroids, indiv_flow, avg_indiv_flow)
        else:
            return (mask, inv_mask, num_labels, labels, stats, centroids, None, None)

    ##
    # @brief Performs optical flow on two images
    # @param curr The current image
    # @param prvs The previous image
    # @param scale The scale param for OF
    # @param pylayers The number of pyramid layers for OF
    # @param winsz The window size for OF
    # @param numit The number of iterations for OF
    # @param polydeg The polynomial degree for OF
    # @param minflowvel The minimum threshold for flow velocity
    # @param drawflow Draw the flow on an image and return it
    # @return mask The total mask of the flow
    # @return inv_mask The inverse of mask
    # @return num_labels The number of regions
    # @return labels The matrix containing the regions
    # @return stats The stats for each region - noted in the code comments
    # @return bgr The blue green red mask
    def findRegionsOF(self, curr, prvs, scale=.9, pylayers=1, winsz=5, numit=1, polydeg=5, minflowvel=.5,
                      drawflow=False):
        hsv = np.zeros_like(curr)
        hsv[..., 1] = 255

        next = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)
        # (prev, next, format, scale, pyramid layers, window size, num iterations, poly degree, std for smoothing, ?)
        # THis one is good w/ minflowvelocity==5
        # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.38, 2, 5, 3, 5, 1.2, 0)

        # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 2, 10, 3, 5, 1.2, 0)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, scale, pylayers, winsz, numit, polydeg, 1.1, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        withflow = curr.copy()
        if drawflow is True:
            withflow = self.draw_flow(curr, flow)

        # Build Detection Mask
        detected = (mag >= minflowvel).astype(int)
        # (mask, inv_mask, num_labels, labels, stats) = findregions(detected)
        return self.findregions(detected, flow), bgr, withflow
