# ---
# jupyter:
#   jupytext_format_version: '1.2'
#   jupytext_formats: ipynb,py
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.4
# ---

# # Development Notebook for IFAT Simulator

from brian2 import *

ifat_mn_eq = '''
    dVmem/dt  = (glm / Cm) * (V_r - Vmem)      : volt
    dtheta/dt = (glt / Ct) * (theta_r - theta) : volt
    
    glm = flm * Cl                             : siemens
    glt = flt * Cl                             : siemens
'''

ifat_syn_eq = '''
    Vsyn = ((W/Cm)*(Em - Vmem)) : volt
    Em         : volt
    W          : farad
    num_events : 1
'''

Cm = Ct = 0.44 * pF
Cl = 0.02 * pF
V_r = 1 * volt
theta_r = 3 * volt
flm = 10 * kHz
flt = 0.2 * MHz
Csm = 0.05 * pF
Cst = 0.00 * pF
theta_max = 5*volt
Eext = 0.66*5 * volt

test = NeuronGroup(6, ifat_mn_eq, threshold='Vmem>theta',
                   reset='''
                           Vmem=V_r
                           theta=theta*(theta>Vmem)+theta_r*(theta<=Vmem)''',
                  method='exact')

spgen = SpikeGeneratorGroup(1,[0],[100*ms])

insyn = Synapses(spgen, test, ifat_syn_eq,
              on_pre='''
                      Vmem_old = Vmem
                      Vmem = Vmem_old + ((W/Cm)*(Em - Vmem_old))
                      theta+=(Cst/Ct)*(Vmem_old - V_r)''',multisynaptic_index='k')

insyn.connect(i=0,j=0,n=100)
insyn.delay='0.2*us+k*0.2*us'
insyn.Em = '0.33*5 * volt'
insyn.W = Csm

vcosyn = Synapses(test,test,ifat_syn_eq,
                  on_pre='''
                      Vmem_old = Vmem
                      Vmem = Vmem_old + ((W/Cm)*(Em - Vmem_old))
                      theta+=(Cst/Ct)*(Vmem_old - V_r)''')
vcosyn.connect('j==((i+1)%6)')
vcosyn.Em = '4*volt'
vcosyn.W = 30 * Csm

# +
import matplotlib.pyplot as plt

# %matplotlib inline
# -

test.Vmem = V_r
test.theta = theta_r

mon  = SpikeMonitor(test)
vmon = StateMonitor(test, 'Vmem', record=True)
tmon = StateMonitor(test,'theta',record=True)

# + {"scrolled": false}
run(1*second)

# + {"scrolled": true}
for i in range(6):
    plot(vmon.t, vmon.Vmem[i],tmon.t, tmon.theta[i])
# -

plot(vmon.t, vmon.Vmem[0],tmon.t, tmon.theta[0]);xlim([0.099,0.11])

# + {"scrolled": true}
for i in range(6):
    plot(vmon.t, vmon.Vmem[i])
    
xlim([0.098,0.102])
# -

plot(mon.t, mon.i,'.'); xlim([0.099,0.11])

insyn

insyn.delay


