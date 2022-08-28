# Code used for testing the consistency between Matlab and Python versions of the metric.

from utils import load_mat_tensor
import torch
import numpy as np
import os

class FovVideoVDP_Testbench():
    def __init__(self, matlab_intermediate_path = "D:\\work\\st_fov_metric\\matlab_intermediate\\"):
        self.tests_passed = 0
        self.tests_failed = 0
        self.matlab_intermediate_path = matlab_intermediate_path

    def print_summary(self):
        total  = self.tests_passed + self.tests_failed
        if total > 0:
            perc = 100.0 * float(self.tests_passed) / float(total)
            print("%d passed of %d total (%0.2f %%)" % (self.tests_passed, total, perc))

    def verify_against_matlab(self, x, tag, device, file = None, tolerance = 0.001, verbose=False, relative=False):
        if file is None:
            mat_tensor = load_mat_tensor(os.path.join(self.matlab_intermediate_path, "%s.mat" % tag), tag, device)
        else:
            mat_tensor = load_mat_tensor(os.path.join(self.matlab_intermediate_path, "%s.mat" % file), tag, device)

        if mat_tensor is None:
            print("Error: cannot find file for tag " + tag)
            return 

        mat_tensor = torch.squeeze(mat_tensor)
        x = torch.squeeze(x)

        assert mat_tensor.shape == x.shape, "Tensor '%s' shape (%s) does not match MATLAB (%s)" % (tag, str(x.shape), str(mat_tensor.shape))

        if relative:
            difftype = "relative"
            diff = torch.abs(x - mat_tensor) / (torch.abs(mat_tensor) + 0.00001)
        else:
            difftype = "absolute"
            diff = torch.abs(x - mat_tensor)

        max_diff = torch.max(diff).cpu().numpy()

        if max_diff > tolerance:
            self.tests_failed += 1
            if len(x.shape)>=1:
                # Tensor
                diff_loc = np.unravel_index([torch.argmax(torch.abs(x - mat_tensor)).cpu().numpy()], x.shape)
                diff_loc = tuple([d[0] for d in diff_loc])
                mean_diff = torch.mean(diff.type(torch.cuda.FloatTensor)).cpu().numpy()
                print("[FAIL] Max %s error for '%s' is %f (> %f) @ %s (mean %f, %f vs %f (matlab))" % (difftype, tag, max_diff, tolerance, str(diff_loc), mean_diff, x[diff_loc], mat_tensor[diff_loc]))
                if verbose:
                    if len(x.shape)==2:
                        for yd in range(-2,3):
                            for xd in range(-2,3):
                                if ((diff_loc[0] + yd) >= 0 and (diff_loc[0] + yd) < x.shape[0] and
                                    (diff_loc[1] + xd) >= 0 and (diff_loc[1] + xd) < x.shape[1]):
                                   print("%0.4f " % x[diff_loc[0] + yd, diff_loc[1] + xd], end='')
                            print('   ', end='')
                            for xd in range(-2,3):
                                if ((diff_loc[0] + yd) >= 0 and (diff_loc[0] + yd) < x.shape[0] and
                                    (diff_loc[1] + xd) >= 0 and (diff_loc[1] + xd) < x.shape[1]):
                                   print("%0.4f " % mat_tensor[diff_loc[0] + yd, diff_loc[1] + xd], end='')
                            print('')
            else:
                # Scalar
                print("[FAIL] Error for '%s' is %f (> %f), %f vs %f (matlab)" % (tag, max_diff, tolerance, x.cpu().item(), mat_tensor.cpu().item()))

            if verbose:
                print("    [Python] " + str(x.flatten()[0:4]))#[...,0:3,0:3])
                print("    [Matlab] " + str(mat_tensor.flatten()[0:4]))#[...,0:3,0:3])
        else:
            self.tests_passed += 1
            if len(x.shape)>=1:
                print("[PASS] Tensor %s matches MATLAB (tolerance %f, max_diff %f)" % (tag, tolerance, max_diff))
            else:
                print("[PASS] Tensor %s matches MATLAB (tolerance %f, max_diff %f,  %f vs %f (matlab))" % (tag, tolerance, max_diff, x.cpu().item(), mat_tensor.cpu().item()))
