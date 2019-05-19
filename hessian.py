import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from numpy import linalg as LA
from torch.autograd import Variable

class Hessian:
    def __init__(self,
                 loader=None,
                 model=None,
                 num_classes=None,
                 hessian_type=None,
                 double=False,
                 class_list=None,
                 vecs=[],
                 vals=[],
                 ):
        
        self.loader                = loader
        self.model                 = model
        self.num_classes           = num_classes
        self.hessian_type          = hessian_type
        self.double                = double
        self.class_list            = class_list
        self.vecs                  = vecs
        self.vals                  = vals
        
        self.device                = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        for i in range(len(self.vecs)):
            self.vecs[i] = self.my_device(self.vecs[i])
            
        f = getattr(nn, 'CrossEntropyLoss')
        self.criterion = f(reduction='sum')
        
        if self.class_list is None:
            if self.num_classes is not None:
                self.class_list = [i for i in range(self.num_classes)]
        
    
    # computes matrix vector multiplication
    # where the matrix is either the Hessian, G or H
    def Hv(self, v):
        Hg = self.my_zero()
        counter = 0
        
        for iter, batch in enumerate(self.loader):
            
            input, target = batch[0], batch[1]
            
            input = input.to(self.device)
            target = target.to(self.device)
            
            input = Variable(input)
            target = Variable(target)
            
            if self.double:
                input = input.double()
                
            f = self.model(input)
            
            loss = self.criterion(f, target)
            
            if self.hessian_type == 'G':
                z = torch.randn(f.shape)
                
                if self.double:
                    z = z.double()
                
                z = z.to(self.device)
                    
                z = Variable(z, requires_grad=True)
                
                # z^T (d f / d theta)
                zT_df_dtheta = torch.autograd.grad(f,
                                                   self.model.parameters(),
                                                   z,
                                                   create_graph=True)
                
                # v^T (z^T (d f / d theta)) / dz
                # (d f / d theta) v
                df_dtheta_v = torch.autograd.grad(zT_df_dtheta,
                                                  z,
                                                  v)
                
                dloss_df = torch.autograd.grad(loss,
                                               f,
                                               create_graph=True)
                
                d2loss_df2_df_dtheta_v = torch.autograd.grad(dloss_df,
                                                             f,
                                                             grad_outputs=df_dtheta_v)
                
                Hg_ = torch.autograd.grad(f,
                                          self.model.parameters(),
                                          grad_outputs=d2loss_df2_df_dtheta_v)
            elif self.hessian_type == 'H':
                dloss_df = torch.autograd.grad(loss,
                                               f)
                
                df_dtheta = torch.autograd.grad(f,
                                                self.model.parameters(),
                                                grad_outputs=dloss_df,
                                                create_graph=True)
                
                df_dtheta[-1].requires_grad = True
                
                Hg_ = torch.autograd.grad(df_dtheta,
                                          self.model.parameters(),
                                          v,
                                          allow_unused=True)
                
                zr = torch.zeros(df_dtheta[-1].shape)
                
                zr = zr.to(self.device)
                
                Hg_ = Hg_[:-1] + (zr,)
            elif self.hessian_type == 'Hessian':
                grad = torch.autograd.grad(loss,
                                           self.model.parameters(),
                                           create_graph=True)
                
                Hg_ = torch.autograd.grad(grad,
                                          self.model.parameters(),
                                          v)
            else:
                raise Exception('Wrong hessian type!')
            
            Hg = self.my_sum(Hg,Hg_)
            
            counter += input.shape[0]
        
        return self.my_div_const(Hg, counter)
    
    
    # computes matrix vector multiplication
    # where the matrix is (H - sum_i val_i vec_i vec_i^T)
    # {val_i}_i and {vec_i}_i are given as input to the class and are usually
    # equal to the top C eigenvalues and eigenvectors
    def mat_vec(self, v):
        Av = self.Hv(v)
        
        for eigvec, eigval in zip(self.vecs, self.vals):
            coeff = eigval * self.my_inner(eigvec, v)
            Av = self.my_sub(Av, self.my_mult_const(eigvec, coeff))
        
        return Av
    
    # compute matrix matrix multiplication by iterating the previous function
    def mat_mat(self, V):
        AV = []
        for v in V:
            AV.append(self.mat_vec(v))
        return AV
    
    # generate a random vector of size #params
    def my_randn(self, skip_cuda=False):
        v_0_l = []
        for param in self.model.parameters():
            Z = torch.randn(param.shape)
            
            if self.double:
                Z = Z.double()
            
            if not skip_cuda:
                Z = Z.to(self.device)
                
            v_0_l.append(Z)
            
        return v_0_l
    
    # the following functions perform basic operations over lists of parameters
    def my_zero(self):
        return [0 for x in self.my_randn()]
    
    def my_sub(self, X, Y):
        return [x-y for x,y in zip(X,Y)]
    
    def my_sum(self, X, Y):
        return [x+y for x,y in zip(X,Y)]
    
    def my_inner(self, X, Y):
        return sum([torch.dot(x.view(-1), y.view(-1)) for x,y in zip(X,Y)])
    
    def my_mult(self, X, Y):
        return [x*y for x,y in zip(X,Y)]
    
    def my_norm(self, X):
        return torch.sqrt(self.my_inner(X,X))
    
    def my_mult_const(self, X, c):
        return [x*c for x in X]
    
    def my_div_const(self, X, c):
        return [x/c for x in X]
    
    def my_len(self):
        X = self.my_randn()
        return sum([x.view(-1).shape[0] for x in X])
    
    def my_data(self, X):
        return [x.data for x in X]
        
    def my_cpu(self, X):
        return [x.cpu() for x in X]
    
    def my_device(self, X):
        return [x.to(self.device) for x in X]
    
    # compute the minimal and maximal eigenvalue of the linear operator mat_vec
    # this is needed for approximating the spectrum using Lanczos
    def compute_lb_ub(self, init_poly_deg):
        ritz_val, S, alp, bet = self.Lanczos(init_poly_deg)
        
        theta_1 = ritz_val[0]
        theta_k = ritz_val[-1]
        
        s_1 = float(bet[-1]) * float(S[-1,0])
        s_k = float(bet[-1]) * float(S[-1,-1])

        t1 = abs(s_1)
        tk = abs(s_k)
        
        lb = theta_1 - t1
        ub = theta_k + tk
        
        return lb, ub
    
    # approximate the spectrum of the linear operator mat_vec
    def LanczosApproxSpec(self, 
                          init_poly_deg,
                          poly_deg,
                          spectrum_margin=0.05,
                          poly_points=1024,
                          log_hessian=False,
                          eps=0,
                          denormalize=True,
                          ):
        
        print('LanczosApproxSpec')
        
        print('Estimating spectrum range')
        
        lb, ub = self.compute_lb_ub(init_poly_deg)
        print('Estimated spectrum range:')
        print('[{}\t{}]'.format(lb, ub))
        
        if log_hessian:
            ub = np.log(max(np.abs(lb),np.abs(ub)) + eps)
            lb = np.log(eps)
            print('log spectrum range:')
            print('[{}\t{}]'.format(lb, ub))
        
        margin = spectrum_margin*(ub - lb)
        
        lb -= margin
        ub += margin
        
        print('Spectrum range after adding margin:')
        print('[{}\t{}]'.format(lb, ub))
        
        c = (lb + ub)/2
        d = (ub - lb)/2
        
        M = poly_deg
        
        LB = -1
        UB = 1
        H = (UB - LB) / (M - 1)
        
        kappa = 1.25
        sigma = H / np.sqrt(8 * np.log(kappa))
        sigma2 = 2 * sigma**2
        
        tol = 1e-08
        width = sigma * np.sqrt(-2.0 * np.log(tol))
        
        aa = LB
        bb = UB
        xdos = np.linspace(aa, bb, poly_points);
        y = np.zeros(poly_points)
        
        print('Approximating spectrum')
        ritz_val, S, _, _ = self.Lanczos(poly_deg)
        
        ritz_val_norm = ritz_val
        
        if log_hessian:
            ritz_val_norm = np.log(np.abs(ritz_val_norm) + eps)
                
        ritz_val_norm = (ritz_val_norm - c) / d
                
        gamma2 = S[0,]**2
                
        diff = np.expand_dims(ritz_val_norm,-1) - np.expand_dims(xdos,0)
        eigval_idx, pts_idx = np.where(np.abs(diff) < width)
        vals = gamma2[eigval_idx] * np.exp(-((xdos[pts_idx] - ritz_val_norm[eigval_idx])**2) / sigma2)
                
        if log_hessian:
            vals = vals / np.abs(ritz_val[eigval_idx] + eps)
                
        np.add.at(y, pts_idx, vals)
        
        scaling = 1.0 / np.sqrt(sigma2 * np.pi)
        y = y*scaling
        
        if denormalize:
            xdos = xdos*d + c
            y = y/d
            
            if log_hessian:
                xdos = np.exp(xdos)
        
        return xdos, y
    
    # M iteratinos of Lanczos on the linear operator mat_vec
    def Lanczos(self, M):
        v = self.my_randn()
        v = self.my_div_const(v, self.my_norm(v))
        
        alp     = torch.zeros(M)
        bet     = torch.zeros(M)
        
        if self.double:
            alp = alp.double()
            bet = bet.double()
        
        alp = alp.to(self.device)
        bet = bet.to(self.device)
        
        v_prev = None
        
        for j in tqdm(range(M)):
            sys.stdout.flush()
            
            v_next = self.mat_vec(v)
            
            if j:
                v_next = self.my_sub(v_next, self.my_mult_const(v_prev,bet[j-1]))
                
            alp[j] = self.my_inner(v_next, v)

            v_next = self.my_sub(v_next, self.my_mult_const(v, alp[j]))
            
            bet[j] = self.my_norm(v_next)
            
            v_next = self.my_div_const(v_next, bet[j])
            
            v_prev = v
            v = v_next
            
        B = np.diag(alp.cpu().numpy()) + np.diag(bet.cpu().numpy()[:-1], k=1) + np.diag(bet.cpu().numpy()[:-1], k=-1)
        ritz_val, S = np.linalg.eigh(B)
        
        return ritz_val, S, alp, bet
    
    # compute top-C eigenvalues and eigenvectors using subspace iteration
    def SubspaceIteration(self, n, iters):
        print('SubspaceIteration')
        
        V = []
        for _ in range(n):
            V.append(self.my_randn())
        
        Q, _ = self.QR(V, n)
        
        for iter in tqdm(range(iters)):
            sys.stdout.flush()
            
            V = self.mat_mat(Q)
            
            eigvals = [self.my_norm(w) for w in V]
            
            Q, _ = self.QR(V, n)
            
        eigval_density = np.ones(len(eigvals)) * 1/len(eigvals)
        
        return Q, eigvals, eigval_density
    
    # QR decomposition, which is needed for subspace iteration
    def QR(self, A, n):
        Q = []
        R = torch.zeros(n,n)
        
        if self.double:
            R = R.double()
            
        R = R.to(self.device)
        
        for j in range(n):
            v = A[j]
            for i in range(j):
                R[i,j] = self.my_inner(Q[i], A[j])
                v = self.my_sub(v, self.my_mult_const(Q[i], R[i,j]))
            
            R[j,j] = self.my_norm(v)
            Q.append(self.my_div_const(v, R[j,j]))
        
        return Q, R
    
    def compute_G_means(self):
        print("Computing {delta_{c,c'}}_{c,c'}")
        
        examples_per_class = 0
        
        delta_ccp = []
        prob_ccp = []
        for c in self.class_list:
            delta_ccp.append([])
            prob_ccp.append([])
            for cp in self.class_list:
                delta_ccp[-1].append(None)
                prob_ccp[-1].append(0)
        
        for idx, batch in enumerate(self.loader, 1):
            print('Iteration: [{}/{}]'.format(idx, len(self.loader)))
            
            sys.stdout.flush()
            
            input, target = batch[0], batch[1]
            
            input = input.to(self.device)
            target = target.to(self.device)
                
            if self.double:
                input = input.double()
            
            input = Variable(input)
            target = Variable(target)
            
            f = self.model(input)
            p = F.softmax(f,dim=1).data
            
            for idx_c, c in enumerate(self.class_list):
                
                idxs = (target == c).nonzero()
                
                if len(idxs) == 0:
                    continue
                
                fc = f[idxs.squeeze(-1),]
                pc = p[idxs.squeeze(-1),]
                
                if idx_c == 0:
                    examples_per_class += fc.shape[0]
                
                for idx_cp, cp in enumerate(self.class_list):
                    w = -pc
                    w[:,cp] = w[:,cp] + 1
                    w *= pc[:,[cp]]
                        
                    J = torch.autograd.grad(fc,
                                            self.model.parameters(),
                                            grad_outputs=w,
                                            retain_graph=True)
                    
                    J = self.my_data(J)
                    
                    J = self.my_cpu(J)
                    
                    if delta_ccp[idx_c][idx_cp] is None:
                        delta_ccp[idx_c][idx_cp] = self.my_zero()
                            
                    delta_ccp[idx_c][idx_cp] = self.my_sum(delta_ccp[idx_c][idx_cp], J)
                    prob_ccp[idx_c][idx_cp] += torch.sum(pc[:,cp])
        
        for idx_c in range(len(self.class_list)):
            for idx_cp in range(len(self.class_list)):
                delta_ccp[idx_c][idx_cp] = [x/prob_ccp[idx_c][idx_cp] for x in delta_ccp[idx_c][idx_cp]]
        
        return delta_ccp, prob_ccp, examples_per_class
    
    def compute_G_decomp(self):

        print('Computing G decomposition')
        
        # delta_{c,c'}
        delta_ccp, prob_ccp, examples_per_class = self.compute_G_means()
        
        C = len(delta_ccp)
        
        delta_ccp_flat = []
        prob_ccp_flat = []
        for c in range(C):
            for c_ in range(C):
                delta_ccp_flat.append(delta_ccp[c][c_])
                prob_ccp_flat.append(prob_ccp[c][c_])
        
        delta_ccp_T_delta_ccp = np.zeros([C**2, C**2])
        for c in tqdm(range(C**2)):
            for c_ in range(C**2):
                delta_ccp_T_delta_ccp[c,c_] = self.my_inner(delta_ccp_flat[c], delta_ccp_flat[c_]) * prob_ccp_flat[c] / examples_per_class / C
        eigval, _ = LA.eig(delta_ccp_T_delta_ccp)
        eigval = np.real(np.abs(sorted(eigval, reverse=True)))
                
        eigval_density  = np.ones(len(eigval)) * 1/len(eigval)
        
        res = {'delta_ccp'       : delta_ccp,
               'eigval'          : eigval,
               'eigval_density'  : eigval_density,
                }
        
        return res
    
    
    # Compute traces of B_{1,c,c'}
    def compute_traces_B1(self,
                          delta_ccp,
                          iters):
        print("Computing traces of B_{1,c,c'}")
        
        trace = torch.zeros(len(self.class_list), len(self.class_list)).to(self.device)
                                                   
        if self.double:
            trace = trace.double()
            
        examples_per_class = 0
        
        for iter in tqdm(range(iters)):
            # delta_ccp is too big to put on GPU
            # compute this part on CPU
            E = self.my_randn(skip_cuda=True)
            
            delta_ccp_E = torch.zeros(len(self.class_list), len(self.class_list))
            if self.double:
                delta_ccp_E = delta_ccp_E.double()
            
            for c in self.class_list:
                for cp in self.class_list:
                    delta_ccp_E[c,cp] = self.my_inner(delta_ccp[c][cp], E)
            
            E = self.my_device(E)
            delta_ccp_E = delta_ccp_E.to(self.device)

            for idx, batch in enumerate(self.loader, 1):
                                                   
                input, target = batch[0], batch[1]
                
                input = input.to(self.device)
                target = target.to(self.device)
                    
                if self.double:
                    input = input.double()
                
                input = Variable(input)
                target = Variable(target)
                
                f = self.model(input)
                    
                p = F.softmax(f,dim=1).data
                
                for idx_c, c in enumerate(self.class_list):
                    
                    idxs = (target == c).nonzero()
                    
                    if len(idxs) == 0:
                        continue
                    
                    fc = f[idxs.squeeze(-1),]
                    pc = p[idxs.squeeze(-1),]
                    
                    z = torch.randn(fc.shape).to(self.device)
                    
                    if self.double:
                        z = z.double()
                    
                    z = Variable(z, requires_grad=True)
                    
                    # z^T (d fc / d theta)
                    zT_dfc_dtheta = torch.autograd.grad(fc,
                                                        self.model.parameters(),
                                                        z,
                                                        create_graph=True)
                    
                    # E^T (z^T (d fc / d theta)) / dz
                    # (d fc / d theta) E
                    dfc_dtheta_E = torch.autograd.grad(zT_dfc_dtheta,
                                                       z,
                                                       E)

                    delta_iccp_E = dfc_dtheta_E[0] - torch.sum(dfc_dtheta_E[0] * pc, dim=1, keepdim=True)
                    diff = delta_iccp_E - delta_ccp_E[c]
                        
                    trace[idx_c] += torch.sum(diff**2 * pc, dim=0)
                    
                    # count number of examples per class
                    if (iter == 0) & (idx_c == 0):
                        examples_per_class += fc.shape[0]
        
        trace = trace / iters
        
        trace = trace / examples_per_class
        trace = trace / len(self.class_list)

        eig = trace / examples_per_class
        
        return trace, eig
    


    