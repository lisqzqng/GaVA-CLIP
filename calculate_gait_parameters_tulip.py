# Load foot prints and calculate gait parameters (7 main) from 'footprint_all.xls'

import os
import os.path as osp

import copy
import pandas as pd
from tqdm import tqdm
import joblib

import numpy as np
from scipy import signal

from collections import defaultdict
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
# Mapping
cm2m = 0.01
minThresh = 0.3 # minimum heel strick interval, 0.3 s
FPS = 30
OFFSET = 0.1 # ratio to extract plausible one-way walking

def get_smpl2_joint_names():
    return [
        'hip',              # 0
        'lhip (SMPL)',      # 1
        'rhip (SMPL)',      # 2
        'spine (SMPL)',     # 3
        'lknee',            # 4
        'rknee',            # 5
        'Spine (H36M)',     # 6
        'lankle',           # 7
        'rankle',           # 8
        'spine2',           # 9
        'leftFoot',         # 10
        'rightFoot',        # 11
        'neck',             # 12
        'lcollar',          # 13
        'rcollar',          # 14
        'Jaw (H36M)',       # 15
        'lshoulder',        # 16
        'rshoulder',        # 17
        'lelbow',           # 18
        'relbow',           # 19
        'lwrist',           # 20
        'rwrist',           # 21
        'leftHand',         # 22
        'rightHand',        # 23
    ]

# SMPL joints with root deplacement
# filenames = [x for x in osp.join('./datasets/orig_tulip/smpl_joint3d') if x.endswith('.npy')]
# skeleton_file = "datasets/wham_skeletons_smpl.json"
skeleton_file = "datasets/wham_skeletons_smpl_465.json"

skeleton_dict = joblib.load(skeleton_file)

varnames = ['vidname', 'speed', 'meansteptime', 'CVsteptime', 'diffsteptime', 'meanstepwidth', 'CVstepwidth', 'diffstepwidth', 'cadence', \
            'mean_minMOS', 'mean_meanMOS']
# initialize the result dictionary
all_file_names = list(set([x.split('_Camera')[0] for x in skeleton_dict.keys()]))
# sort the elements in the skeleton dictionary
skeleton_dict = {k: v for k, v in sorted(skeleton_dict.items(), key=lambda item: item[0].split('_')[1])}
assert len(all_file_names)==10
check_dict = {x: 0 for x in all_file_names}
Result = defaultdict(list)

plt.figure()
for idx, (fn, data) in enumerate(skeleton_dict.items()):
    to_check = False
    print(fn)
    # if not Camera in fn: continue
    gscore = data['gait_score']

    kinematic = data['joints3D']
    num = kinematic.shape[0]
    # put the skeleton on the ground
    lowest_joint = np.argmin(kinematic[:,:,1], axis=1)
    kinematic[...,1] -= kinematic[np.arange(num), None, lowest_joint, 1]
    Time = np.arange(num) / FPS
    # get the pelvis/foot/toe
    pelv = kinematic[:, 0]
    Rhip = kinematic[:, 2]
    Rfoot = kinematic[:, 11]
    Lhip = kinematic[:, 1]
    Lfoot = kinematic[:, 10]

    # ======> Step 1. filter the sensor signals with butterworth filter <====== #
    order = 2
    cutoff = 4 # cutoff frequency in Hz
    b, a = signal.butter(order, cutoff / (FPS / 2))
    _b, _a = signal.butter(order, 0.5*cutoff / (FPS / 2))
    pelvx = signal.filtfilt(b, a, pelv[:, 0])
    pelvy = signal.filtfilt(b, a, pelv[:, 1])
    pelvz = signal.filtfilt(b, a, pelv[:, 2])

    Rhipx = signal.filtfilt(b, a, Rhip[:, 0])
    Rhipy = signal.filtfilt(b, a, Rhip[:, 1])
    Rhipz = signal.filtfilt(b, a, Rhip[:, 2])
    Rfootx = signal.filtfilt(_b, _a, Rfoot[:, 0])
    Rfooty = signal.filtfilt(_b, _a, Rfoot[:, 1])
    Rfootz = signal.filtfilt(_b, _a, Rfoot[:, 2])

    Lhipx = signal.filtfilt(b, a, Lhip[:, 0])
    Lhipy = signal.filtfilt(b, a, Lhip[:, 1])
    Lhipz = signal.filtfilt(b, a, Lhip[:, 2])
    Lfootx = signal.filtfilt(_b, _a, Lfoot[:, 0])
    Lfooty = signal.filtfilt(_b, _a, Lfoot[:, 1])
    Lfootz = signal.filtfilt(_b, _a, Lfoot[:, 2])

    # ======> Step 2. cut trajectory into sub-sequences <====== #
    # get the extremities of the trajectory
    xmax = np.max(pelvx)
    xmin = np.min(pelvx)
    maxthresh = (1-OFFSET) * (xmax-xmin) + xmin
    minthresh = OFFSET * (xmax-xmin) + xmin
    # find the intersection of the pelvis x signal w.r.t. the threshold
    idx = np.where((pelvx < maxthresh) & (pelvx > minthresh))[0]

    # =====> Step 2.1 prepare quantities for MOS calculation  <===== #
    # leg length
    # --> right leg
    RLeg = np.vstack((Rhipx, Rhipy, Rhipz)).T -  np.vstack((Rfootx, Rfooty, Rfootz)).T
    LLeg = np.vstack((Lhipx, Lhipy, Lhipz)).T - np.vstack((Lfootx, Lfooty, Lfootz)).T
    Leglength = np.max(np.sqrt((RLeg**2).sum(axis=1)))*0.5 + \
        np.max(np.sqrt((LLeg**2).sum(axis=1)))*0.5

    # XCOM, COM: center of mass
    COM = np.vstack([pelvx, pelvy, pelvz]).T
    omega = np.sqrt(9.81 / Leglength) # angular velocity
    VCOM = np.diff(COM, axis=0)
    VCOM = np.append(VCOM, [VCOM[-1]], axis=0)

    XCOMML = COM + (VCOM / omega)  # extrapolated center of mass in ML

    # separating right and left steps for XCOM and calculating MOS
    Rfoot = np.vstack((Rfootx, Rfooty, Rfootz)).T
    Lfoot = np.vstack((Lfootx, Lfooty, Lfootz)).T
    RMOS0 = np.sqrt(np.sum((XCOMML - Rfoot)**2, axis=-1))
    LMOS0 = np.sqrt(np.sum((XCOMML - Lfoot)**2, axis=-1))

    vidname = osp.basename(fn).split('.')[0]
    # =====> Step 3.1 detect Heel Strike Moments <===== #
    RH = signal.argrelextrema(Rfooty, np.less)[0]
    LH = signal.argrelextrema(Lfooty, np.less)[0]

    # Post-process Heel strikes
    try:
        if RH[0] < LH[0]:
            refHS = copy.deepcopy(LH) # insert RH into LH
            reffoot = Lfoot[:,1]
            interpHS = copy.deepcopy(RH)
            interpfoot = Rfoot[:,1]
        else:
            refHS = copy.deepcopy(RH) # insert LH into RH
            reffoot = Rfoot[:,1]
            interpHS = copy.deepcopy(LH)
            interpfoot = Lfoot[:,1]
    except:
        print(f"Error in {fn}")
        continue
    
    _refHS, _interpHS = [], []
    # Inserting interHS
    # make sure the interpHS is in between refHS
    for i, rh in enumerate(refHS):
        if i>0:
            if rh-refHS[i-1]<minThresh*FPS:
                continue
        to_select_id = np.where(interpHS<rh)[0]
        to_select = interpHS[to_select_id]
        if len(to_select)==0:
            continue
        elif len(to_select)==1:
            _refHS.append(rh)
            _interpHS.append(to_select[0])
        else:
            indx = np.argmin(interpfoot[to_select])
            _refHS.append(rh)
            _interpHS.append(to_select[indx])
        interpHS = interpHS[to_select_id[-1]+1:]
    to_select = interpHS[interpHS>rh]
    if len(to_select)==1:
        _interpHS.append(to_select[0])
    elif len(to_select)>1:
        indx = np.argmin(interpfoot[to_select])
        _interpHS.append(to_select[indx])

    if RH[0] < LH[0]:
        locslheels = _refHS
        locsrheels = _interpHS
    else:
        locslheels = _interpHS
        locsrheels = _refHS
    # add the last frame as the last heel strike for short video
    assert len(locslheels)*len(locsrheels)>0, f"No heel stri<e detected in {vidname}"

    # Plotting HS points to check
    # Right
    if False:
        plt.figure()
        plt.plot(Rfooty, label='RAnkle')
        plt.plot(locsrheels, Rfooty[locsrheels], 'r^', label='RHS')
        plt.ylabel('RAnkle')
        plt.xlabel('Time')
        plt.legend()
        plt.show()

        A = int(input('Are Right HS points correct? (input 1 for Yes or 2 for No): '))
        while A != 1 and A != 2:
            A = int(input('Are Right HS points correct? (input 1 for Yes or 2 for No): '))
        if A == 2:
            nr = input('Type the ID of wrong position of RHS points (e.g. 2): ')
            to_remove = [int(x) for x in nr.split(' ')]
            # for i in range(len(locsrheels)):
            #     nr = int(input('Type the ID of wrong position of RHS points (e.g. 2): '))
            #     r = int(input('Type the correct value of wrong position of RHS points (e.g. 33): '))
            #     if nr<0:break
            #     if r<0:
            #         to_remove.append(nr)
            #     else:
            #         locsrheels[nr] = r
            #     B = int(input('Is there any more? (input 1 for Yes or 2 for No): '))
            #     if B == 2:
            #         break
            locsrheels = [locsrheels[i] for i in range(len(locsrheels)) if i not in to_remove]

        # Left
        plt.figure()
        plt.plot(Lfooty, label='LAnkle')
        plt.plot(locslheels, Lfooty[locslheels], 'r^', label='LHS')
        plt.ylabel('LAnkle')
        plt.xlabel('Time')
        plt.legend()
        plt.show()

        C = int(input('Are Left HS points correct? (input 1 for Yes or 2 for No): '))
        while C != 1 and C != 2:
            C = int(input('Are Left HS points correct? (input 1 for Yes or 2 for No): '))
        if C == 2:
            nl = input('Type the ID of wrong position of LHS points (e.g. 2): ')
            to_remove = [int(x) for x in nl.split(' ')]
            # for i in range(len(locslheels)):
            #     nl = int(input('Type the ID of wrong position of LHS points (e.g. 2): '))
            #     l = int(input('Type correct value of wrong position of LHS points (e.g. 33): '))
            #     if nl<0:break
            #     if l<0:
            #         to_remove.append(nl)
            #     else:
            #         locslheels[nl] = l
            #     D = int(input('Is there any more? (input 1 for Yes or 2 for No): '))
            #     if D == 2:
            #         break
            locslheels = [locslheels[i] for i in range(len(locslheels)) if i not in to_remove]


    RHS = locsrheels
    LHS = locslheels

    if False:
        # Plotting HS points to check
        # Right
        plt.clf()
        plt.subplot(1,2,1)
        plt.plot(Rfooty, label='RFoot')
        plt.plot(locsrheels, Rfooty[locsrheels], 'r^', label='RHS')
        plt.ylabel('RFoot')
        plt.xlabel('Time')
        plt.legend()

        # Left
        plt.subplot(1,2,2)
        plt.plot(Lfooty, label='LFoot')
        plt.plot(locslheels, Lfooty[locslheels], 'r^', label='LHS')
        plt.ylabel('LFoot')
        plt.xlabel('Time')
        plt.legend()
        plt.show()
        # if plt.waitforbuttonpress(10):
        #     pass
        #     # C = int(input('Are Left HS points correct? (input 1 for Yes or 2 for No): '))
        #     # while C != 1 and C != 2:
        #     #     C = int(input('Are Left HS points correct? (input 1 for Yes or 2 for No): '))
        #     # if C == 2:
        # else:
        #     print(f"Error in file {fn} !!")
        #     continue

    # RHS = locsrheels
    # LHS = locslheels

    lheels = [Lfoot[lh] for lh in locslheels]
    rheels = [Rfoot[rh] for rh in locsrheels]

    # =====> Step 3.2 Calculate step time & step width from heel strikes & Margin of Stability <===== #
    Rsteptime = []
    Lsteptime = []
    Rstepwidth = []
    Lstepwidth = []
    Rspeed, Lspeed = [], []
    # Process MoS along with steps
    minRMOS = []
    meanRMOS = []
    minLMOS = []
    meanLMOS = []

    try:    
        for idx, (rhs, lhs) in enumerate(zip(RHS, LHS)):
            if rhs>lhs:
                # right step
                Rsteptime.append((rhs-lhs)*1/FPS)
                Rstepwidth.append(np.sqrt(np.sum((rheels[idx] - lheels[idx])**2)))
                Rspeed.append(Rstepwidth[-1] / Rsteptime[-1])
                RMOS = RMOS0[lhs:rhs]
                minRMOS.append(np.nanmin(np.abs(RMOS)))
                meanRMOS.append(np.nanmean(np.abs(RMOS)))
                if idx>0:
                    # left step
                    Lsteptime.append((lhs-RHS[idx-1])*1/FPS)
                    Lstepwidth.append(np.sqrt(np.sum((lheels[idx] - rheels[idx-1])**2)))
                    Lspeed.append(Lstepwidth[-1] / Lsteptime[-1])
                    LMOS = LMOS0[RHS[idx-1]:lhs]
                    minLMOS.append(np.nanmin(np.abs(LMOS)))
                    meanLMOS.append(np.nanmean(np.abs(LMOS)))
            else:
                # left step
                Lsteptime.append((lhs-rhs)*1/FPS)
                Lstepwidth.append(np.sqrt(np.sum((lheels[idx] - rheels[idx])**2)))
                Lspeed.append(Lstepwidth[-1] / Lsteptime[-1])
                LMOS = LMOS0[rhs:lhs]
                minLMOS.append(np.nanmin(np.abs(LMOS)))
                meanLMOS.append(np.nanmean(np.abs(LMOS)))
                if idx>0:
                    # right step
                    Rsteptime.append((rhs-LHS[idx-1])*1/FPS)
                    Rstepwidth.append(np.sqrt(np.sum((rheels[idx] - lheels[idx-1])**2)))
                    Rspeed.append(Rstepwidth[-1] / Rsteptime[-1])
                    RMOS = RMOS0[LHS[idx-1]:rhs]
                    minRMOS.append(np.nanmin(np.abs(RMOS)))
                    meanRMOS.append(np.nanmean(np.abs(RMOS)))
    except:
        print(f"Error in {fn}")
        continue

    if len(RHS)>len(LHS):
        Rsteptime.append((RHS[idx+1]-LHS[idx])*1/FPS)
        Rstepwidth.append(np.sqrt(np.sum((rheels[idx+1] - lheels[idx])**2)))
        Rspeed.append(Rstepwidth[-1] / Rsteptime[-1])
        RMOS = RMOS0[LHS[idx]:RHS[idx+1]]
        minRMOS.append(np.nanmin(np.abs(RMOS)))
        meanRMOS.append(np.nanmean(np.abs(RMOS)))
    elif len(RHS)<len(LHS):
        Lsteptime.append((LHS[idx+1]-RHS[idx])*1/FPS)
        Lstepwidth.append(np.sqrt(np.sum((lheels[idx+1] - rheels[idx])**2)))
        Lspeed.append(Lstepwidth[-1] / Lsteptime[-1])
        LMOS = LMOS0[RHS[idx]:LHS[idx+1]]
        minLMOS.append(np.nanmin(np.abs(LMOS)))
        meanLMOS.append(np.nanmean(np.abs(LMOS)))


    steptime = np.concatenate((Rsteptime, Lsteptime))
    stepwidth = np.concatenate((Rstepwidth, Lstepwidth))
    Rspeed, Lspeed = np.array(Rspeed), np.array(Lspeed)

    # speed
    speed = np.mean(np.concatenate((Rspeed, Lspeed)))
    # step time
    meansteptime = np.mean(steptime)
    if len(Rsteptime) * len(Lsteptime)==0:
        diffsteptime = 'NA'
        cv_speed = 'NA'
        CVsteptime = 'NA'
    else:
        cv_speed = np.std(np.concatenate((Rspeed, Lspeed))) / speed * 100.
        sdsteptime = np.std(steptime)
        CVsteptime = sdsteptime / meansteptime * 100.
        diffsteptime = abs(np.array(Lsteptime).mean() - np.array(Rsteptime).mean())
    # step width
    meanstepwidth = np.mean(stepwidth)
    if len(Rstepwidth) * len(Lstepwidth)==0:
        diffstepwidth = 'NA'
        CVstepwidth = 'NA'
    else:
        sdstepwidth = np.std(stepwidth)
        CVstepwidth = sdstepwidth / meanstepwidth * 100.
        diffstepwidth = np.array(Rstepwidth).mean() - np.array(Lstepwidth).mean()
        diffstepwidth = abs(diffstepwidth)
    cadence = 60 / meansteptime

    # =====> Step 3.3 Margin of stability (MOS) in ML direction <===== #

    # calculating Margin Of Stability
    minMOS = np.concatenate((minRMOS, minLMOS))
    mean_minMOS = np.mean(minMOS)
    sd_minMOS = np.std(minMOS)

    meanMOS = np.concatenate((meanRMOS, meanLMOS))
    mean_meanMOS = np.mean(meanMOS)

    # calculate gait parameters
    speed = np.mean(np.concatenate((Rspeed, Lspeed)))
    meansteptime = np.mean(steptime)
    sdsteptime = np.std(steptime)
    CVsteptime = sdsteptime / meansteptime
    diffsteptime = np.array(Rsteptime).mean() - np.array(Lsteptime).mean()
    meanstepwidth = np.mean(stepwidth)
    sdstepwidth = np.std(stepwidth)
    CVstepwidth = sdstepwidth / meanstepwidth
    diffstepwidth = np.array(Rstepwidth).mean() - np.array(Lstepwidth).mean()
    cadence = 60 / meansteptime

    # =====> Step 3.3 Margin of stability (MOS) in ML direction <===== #

    # calculating Margin Of Stability
    minMOS = np.concatenate((minRMOS, minLMOS))
    mean_minMOS = np.mean(minMOS)
    sd_minMOS = np.std(minMOS)

    meanMOS = np.concatenate((meanRMOS, meanLMOS))
    mean_meanMOS = np.mean(meanMOS)
    sd_meanMOS = np.std(meanMOS)

    # add the result to the dictionary
    Result['vidname'].append(vidname)
    # Result['checked'].append(1 if to_check else 0)
    Result['diag'].append(data['diag'])
    Result['updrs'].append(gscore)
    Result['leglength'].append(Leglength)
    Result['cadence'].append(cadence)
    Result['speed'].append(speed)
    Result['meanstepwidth'].append(meanstepwidth)
    Result['meansteptime'].append(meansteptime)
    Result['diffstepwidth'].append(abs(diffstepwidth))
    Result['diffsteptime'].append(abs(diffsteptime))
    Result['CVstepwidth'].append(CVstepwidth)
    Result['CVsteptime'].append(CVsteptime)
    Result['mean_minMOS'].append(mean_minMOS)
    Result['mean_meanMOS'].append(mean_meanMOS)

# Exporting result
# Write the result to an Excel file
df = pd.DataFrame(Result)
df.to_excel(f'./data/tulip_basic_gparams.xlsx', sheet_name='part1', index=False)