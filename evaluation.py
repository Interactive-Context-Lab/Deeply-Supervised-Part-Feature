import argparse
import os
import csv
# import time
import numpy as np

from utils.distance import compute_dist, normalize
from utils.re_ranking import re_ranking

######################################################################
# Options
# --------
parser = argparse.ArgumentParser()

parser.add_argument('--feature_dir', type=str, default='features/market_6part_atten/')
parser.add_argument('--result_dir', type=str, default='result/market_6part_atten/')
parser.add_argument('--dataset', type=str, default='market', choices=['market', 'duke', 'all', 'EE3f'])
parser.add_argument('--iteration', type=int, default=10)
# test parameter
parser.add_argument('--logfile', type=bool, default=True)
parser.add_argument('--normalize_feat', type=bool, default=True)
parser.add_argument('--rerank', type=bool, default=False)
args = parser.parse_args()

def plotcurve(filename, TTR, FTR):
    import matplotlib.pyplot as plt
    plt.figure(1)  # 创建图表1
    plt.title('TTR/FTR Curve')  # give plot a title
    plt.xlabel('FTR')  # make axis labels
    plt.ylabel('TTR')
    plt.xscale('log')
    plt.grid()
    # plt.xlim(1e-5, 1)
    plt.figure(1)
    plt.plot(FTR, TTR)
    # plt.show()
    plt.savefig('%s.png' % filename)
    plt.close()

def readfeatures(featfile, normalize_feat=True):
    print("Read %s" % featfile)
    feat_info = np.genfromtxt(featfile, delimiter=',', skip_header=1)
    print('shape of feature_info: ', np.shape(feat_info[:, 3:]))

    if normalize_feat:
        feat_info[:, 3:] = normalize(feat_info[:, 3:], axis=1)
    return feat_info

def SV(feature_path, result_path, plot=False):
    queryfeatpath = feature_path + '/query.csv'
    query = readfeatures(queryfeatpath, args.normalize_feat)
    q_feat = query[:, 3:]
    q_ids = query[:, 1].astype(np.int)

    galleryfeatpath = feature_path + '/gallery.csv'
    gallery = readfeatures(galleryfeatpath, args.normalize_feat)

    g_feat = gallery[:, 3:]
    g_ids = gallery[:, 1].astype(np.int)
    resultfile = result_path
    if not os.path.exists(os.path.dirname(resultfile)):
        os.makedirs(os.path.dirname(resultfile))
    f = open(resultfile, "w", newline='')
    w = csv.writer(f)
    head = ["Threshold", "TTR", "FTR"]
    w.writerow(head)

    ingalleryID = list(np.unique(g_ids))

    q_g_dist = compute_dist(q_feat, g_feat, type='euclidean')
    if args.rerank:
        ##########################
        # Re-ranked Single Query #
        ##########################
        print('Computing Re-ranking distance...')
        # st = time.time()

        # gallery-gallery distance
        g_g_dist = compute_dist(g_feat, g_feat, type='cosine')
        # print('shape of g_g_dist: ', np.shape(g_g_dist))

        for i in range(len(q_feat)):
            temp_q_feat = np.expand_dims(q_feat[i], axis=0)
            q_q_dist = compute_dist(temp_q_feat, temp_q_feat, type='euclidean')
            # print('shape of q_q_dist: ', np.shape(q_q_dist))

            # re-ranked query-gallery distance
            q_g_dist[i] = re_ranking(np.expand_dims(q_g_dist[i], axis=0), q_q_dist, g_g_dist, k1=10, k2=5)
            # query-query distance

    # similar_idx = np.argmin(q_g_dist, axis=1)
    similar_dist = np.min(q_g_dist, axis=1)
    # ed = time.time()
    # print("%f" % (ed-st))
    TTR_list = []
    FTR_list = []
    TTR_to_FTR = np.zeros(6, dtype=np.float)

    print("Start SV evaluating: {}".format(os.path.dirname(queryfeatpath)))
    for t in range(0, 2000):
        threshold = t / 1000.0
        match_mask = similar_dist <= threshold
        # ed = time.clock()
        # print("{:.6f}".format(ed - st))
        inidmask = np.in1d(q_ids, ingalleryID)
        TQ = float(len(q_ids[inidmask]))
        NTQ = float(len(q_ids[np.logical_not(inidmask)]))

        matched_q = q_ids[match_mask]
        matached_inidmask = np.in1d(matched_q, ingalleryID)
        TTQ = float(list(matached_inidmask).count(True))
        FNTQ = float(list(matached_inidmask).count(False))

        TTR = TTQ / TQ * 100.0
        FTR = FNTQ / NTQ * 100.0

        # print("Threshold: {}, TTR: {:.3f}, FTR: {:.3f}".format(threshold, TTR, FTR))
        w.writerow([threshold, TTR, FTR])
        f.flush()
        TTR_list.append(TTR)
        FTR_list.append(FTR)

        if FTR <= 0.1:
            TTR_to_FTR[0] = TTR
        elif FTR <= 1.0:
            TTR_to_FTR[1] = TTR
        elif FTR <= 5.0:
            TTR_to_FTR[2] = TTR
        elif FTR <= 10.0:
            TTR_to_FTR[3] = TTR
        elif FTR <= 20.0:
            TTR_to_FTR[4] = TTR
        elif FTR <= 30.0:
            TTR_to_FTR[5] = TTR

    f.close()

    if plot:
        graphname = 'gallery_' + args.dataset + "_SV_%d" % (i)
        plotcurve(graphname, TTR_list, FTR_list)
    print()
    print("Test SV: %s" % (resultfile))
    print("==================================")
    print("FTR:\t0.1,\t1,\t5,\t10,\t20,\t30")
    print("TTR:\t{:.2f},\t{:.2f},\t{:.2f}, \t{:.2f},\t{:.2f},\t{:.2f}\n\n".format(*TTR_to_FTR.tolist()))

    return TTR_to_FTR


def IV(feature_path, result_path, plot=False):
    queryfeatpath = feature_path + '/query.csv'
    query = readfeatures(queryfeatpath, args.normalize_feat)
    q_feat = query[:, 3:]
    q_ids = query[:, 1].astype(np.int)

    galleryfeatpath = feature_path + '/gallery.csv'
    gallery = readfeatures(galleryfeatpath, args.normalize_feat)

    g_feat = gallery[:, 3:]
    g_ids = gallery[:, 1].astype(np.int)
    resultfile = result_path

    if not os.path.exists(os.path.dirname(resultfile)):
        os.makedirs(os.path.dirname(resultfile))
    f = open(resultfile, "w", newline='')
    w = csv.writer(f)
    head = ["Threshold", "TTR", "FTR"]
    w.writerow(head)

    ingalleryID = list(np.unique(g_ids))
    q_g_dist = compute_dist(q_feat, g_feat, type='euclidean')
    if args.rerank:
        ##########################
        # Re-ranked Single Query #
        ##########################
        print('Computing Re-ranking distance...')
        # st = time.time()

        # gallery-gallery distance
        g_g_dist = compute_dist(g_feat, g_feat, type='euclidean')
        # print('shape of g_g_dist: ', np.shape(g_g_dist))

        for i in range(len(q_feat)):
            temp_q_feat = np.expand_dims(q_feat[i], axis=0)
            q_q_dist = compute_dist(temp_q_feat, temp_q_feat, type='euclidean')
            # print('shape of q_q_dist: ', np.shape(q_q_dist))

            # re-ranked query-gallery distance
            q_g_dist[i] = re_ranking(np.expand_dims(q_g_dist[i], axis=0), q_q_dist, g_g_dist, k1=10, k2=5)
            # query-query distance

    TTR_list = []
    FTR_list = []
    TTR_to_FTR = np.zeros(6, dtype=np.float)

    print("Start IV evaluating: {}".format(os.path.dirname(queryfeatpath)))
    for t in range(0, 2000):
        threshold = t / 1000.0
        iv_TTR = np.zeros(len(ingalleryID), dtype=np.float)
        iv_FTR = np.zeros(len(ingalleryID), dtype=np.float)
        for j, id in enumerate(ingalleryID):

            g_mask = g_ids == id
            onlyg_q_dist = q_g_dist[:, g_mask]
            similar_dist = np.min(onlyg_q_dist, axis=1)
            similar_idx = np.argmin(onlyg_q_dist, axis=1)

            ingallerymask = q_ids == id
            TQ = float(list(ingallerymask).count(True))
            NTQ = float(list(ingallerymask).count(False))

            ing_dist = similar_dist[ingallerymask]
            match_dist = ing_dist <= threshold

            TTQ = float(list(match_dist).count(True))

            not_ing_dist = similar_dist[np.logical_not(ingallerymask)]
            # not_ing_idx = similar_idx[np.logical_not(ingallerymask)]

            not_ing_match_dist = not_ing_dist <= threshold
            # not_ing_match_dist = np.bitwise_and(not_ing_dist <= threshold, not_ing_pre_idx == id)
            FNTQ = float(list(not_ing_match_dist).count(True))

            iv_TTR[j] = TTQ / TQ * 100.0
            iv_FTR[j] = FNTQ / NTQ * 100.0

        TTR = np.mean(iv_TTR)
        FTR = np.mean(iv_FTR)
        TTR_list.append(TTR)
        FTR_list.append(FTR)
        # print("Threshold: {}, TTR: {:.3f}, FTR: {:.3f}".format(threshold, TTR, FTR))
        w.writerow([threshold, TTR, FTR])
        f.flush()

        if FTR <= 0.1:
            TTR_to_FTR[0] = TTR
        elif FTR <= 1.0:
            TTR_to_FTR[1] = TTR
        elif FTR <= 5.0:
            TTR_to_FTR[2] = TTR
        elif FTR <= 10.0:
            TTR_to_FTR[3] = TTR
        elif FTR <= 20.0:
            TTR_to_FTR[4] = TTR
        elif FTR <= 30.0:
            TTR_to_FTR[5] = TTR

    f.close()

    if plot:
        graphname = 'gallery_' + args.dataset + "_IV_%d" % (i)
        plotcurve(graphname, TTR_list, FTR_list)

    print()
    print("Test IV: %s" % (resultfile))
    print("==================================")
    print("FTR:\t0.1,\t1,\t5,\t10,\t20,\t30")
    print("TTR:\t{:.2f},\t{:.2f},\t{:.2f}, \t{:.2f},\t{:.2f},\t{:.2f}\n\n".format(*TTR_to_FTR.tolist()))

    return TTR_to_FTR#, np.asarray(TTR_list), np.asarray(FTR_list)

if __name__ == "__main__":

    feature_path = "{}/{}/test{}/"
    result_path = "{}/{}/test{}/result_{}.csv"

    SV_result = np.zeros((args.iteration, 6), dtype=np.float)
    IV_result = np.zeros((args.iteration, 6), dtype=np.float)
    SV_TTR_list = []
    SV_FTR_list = []
    for i in range(args.iteration):
        SV_result[i] = SV(feature_path.format(args.feature_dir, args.dataset, i),
                            result_path.format(args.result_dir, args.dataset, i, "SV"))

        IV_result[i] = IV(feature_path.format(args.feature_dir, args.dataset, i),
                        result_path.format(args.result_dir, args.dataset, i, "IV"))

    mean_result = args.result_dir + "mean_SV_IV_result.csv"
    with open(mean_result, "w") as f:
        w = csv.writer(f)
        w.writerow(["SV", "0.1%", "1%", "5%", "10%", "20%", "30%", "", "IV", "0.1%", "1%", "5%", "10%", "20%", "30%"])
        for i in range(args.iteration):
            writedata = []
            writedata.extend(["SV %d" %i])
            writedata.extend(SV_result[i].tolist())
            writedata.extend([""])
            writedata.extend(["IV %d" %i])
            writedata.extend(IV_result[i].tolist())
            w.writerow(writedata)

        writedata = []
        writedata.extend(["mean SV"])
        writedata.extend(np.mean(SV_result, axis=0).tolist())
        writedata.extend([""])
        writedata.extend(["mean IV"])
        writedata.extend(np.mean(IV_result, axis=0).tolist())
        w.writerow(writedata)

    # plotline("sv_picture.jpg", np.mean(SV_result, axis=0), [0.1, 1.0, 5.0, 10.0, 20.0, 30.0])


