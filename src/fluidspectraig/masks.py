# MIT License

# Copyright (c) 2023 louity

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn.functional as F

class Masks:
    def __init__(self, mask):
        mtype = mask.dtype
        self.q = mask.reshape((1,)*(4-len(mask.shape)) + mask.shape)
        self.u = F.avg_pool2d(self.q, (2,1), stride=(1,1), padding=(1,0)) > 3/4
        self.v = F.avg_pool2d(self.q, (1,2), stride=(1,1), padding=(0,1)) > 3/4
        self.psi = F.avg_pool2d(self.q, (2,2), stride=(1,1), padding=(1,1)) > 7/8

        self.not_q = torch.logical_not(self.q.type(torch.bool))
        self.not_u = torch.logical_not(self.u)
        self.not_v = torch.logical_not(self.v)
        self.not_psi = torch.logical_not(self.psi)

        self.psi_irrbound_xids, self.psi_irrbound_yids = torch.where(
                torch.logical_and(self.not_psi[0,0,1:-1,1:-1],
                F.avg_pool2d(self.psi.type(mtype), (3,3), stride=(1,1))[0,0] > 1/18)
            )

        self.psi_distbound1 = torch.logical_and(
            F.avg_pool2d(self.psi.type(mtype), (3,3), stride=(1,1), padding=(1,1)) < 17/18,
            self.psi)

        self.omega_inside = torch.logical_and(
                torch.logical_not(self.psi_distbound1),
                self.psi)

        self.u_bound = torch.logical_and(self.not_u[0,0,:,:],
                                        F.avg_pool2d(self.u.type(mtype), (3,3), stride=(1,1),padding=(1,1))[0,0] > 1/18)

        self.v_bound = torch.logical_and(self.not_v[0,0,:,:],
                                        F.avg_pool2d(self.v.type(mtype), (3,3), stride=(1,1),padding=(1,1))[0,0] > 1/18)
        # To do : need to verify q_irrbound_[x,y]ids
        # The irregular points for the laplacian with
        # neumann boundary conditions are the points that
        # straddle the boundary; recall that we are using the tracer points
        # for the pv locations and the boundary points lie on tracer cell edges.
        # self.q_irrbound_xids, self.q_irrbound_yids = torch.where(
        #         torch.logical_and(self.not_q[0,0,1:-1,1:-1],
        #                           F.avg_pool2d(self.q.type(mtype), (3,3), stride=(1,1))[0,0] > 1/18)
        #     )

        # This gets us cells that are in the boundary (not_q = 1 in the boundary) and right on the border with
        # the interior
        #self.q_bound_b = torch.logical_and(self.not_q[0,0,:,:],
        #                               F.avg_pool2d(self.q.type(mtype), (3,3), stride=(1,1),padding=(1,1))[0,0] > 1/18)

        # This gets us cells that are in the interior (q = 1 in the interior) and right on the border with the boundary
        self.q_bound = torch.logical_and(self.q[0,0,:,:],
                                        F.avg_pool2d(self.not_q.type(mtype), (3,3), stride=(1,1),padding=(1,1))[0,0] > 1/18)
        
        self.q_distbound1 = torch.logical_and(
            F.avg_pool2d(self.q.type(mtype), (3,3), stride=(1,1), padding=(1,1)) < 17/18,
            self.q)

        self.u_distbound1 = torch.logical_and(
            F.avg_pool2d(self.u.type(mtype), (3,1), stride=(1,1), padding=(1,0)) < 5/6,
            self.u)
        self.u_distbound2plus = torch.logical_and(
            torch.logical_not(self.u_distbound1), self.u)
        self.u_distbound2 = torch.logical_and(
            F.avg_pool2d(self.u.type(mtype), (5,1), stride=(1,1), padding=(2,0)) < 9/10,
            self.u_distbound2plus)
        self.u_distbound3plus = torch.logical_and(
            torch.logical_not(self.u_distbound2), self.u_distbound2plus)

        self.v_distbound1 = torch.logical_and(
            F.avg_pool2d(self.v.type(mtype), (1,3), stride=(1,1), padding=(0,1)) < 5/6,
            self.v)
        self.v_distbound2plus = torch.logical_and(
            torch.logical_not(self.v_distbound1), self.v)
        self.v_distbound2 = torch.logical_and(
            F.avg_pool2d(self.v.type(mtype), (1,5), stride=(1,1), padding=(0,2)) < 9/10,
            self.v_distbound2plus)
        self.v_distbound3plus = torch.logical_and(
            torch.logical_not(self.v_distbound2), self.v_distbound2plus)


        self.q = self.q.type(mtype)
        self.u = self.u.type(mtype)
        self.v = self.v.type(mtype)
        self.psi = self.psi.type(mtype)
        self.not_q = self.not_q.type(mtype)
        self.not_u = self.not_u.type(mtype)
        self.not_v = self.not_v.type(mtype)
        self.not_psi = self.not_psi.type(mtype)
        self.omega_inside = self.omega_inside.type(mtype)

        self.psi_distbound1 = self.psi_distbound1.type(mtype)
        self.q_distbound1 = self.q_distbound1.type(mtype)

        self.u_distbound1 = self.u_distbound1.type(mtype)
        self.u_distbound2 = self.u_distbound2.type(mtype)
        self.u_distbound2plus = self.u_distbound2plus.type(mtype)
        self.u_distbound3plus = self.u_distbound3plus.type(mtype)

        self.v_distbound1 = self.v_distbound1.type(mtype)
        self.v_distbound2 = self.v_distbound2.type(mtype)
        self.v_distbound2plus = self.v_distbound2plus.type(mtype)
        self.v_distbound3plus = self.v_distbound3plus.type(mtype)