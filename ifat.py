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
#     version: 3.6.0
# ---

# # Development Notebook for IFAT Simulator

# +
from brian2 import *
import matplotlib.pyplot as plt

# %matplotlib inline
# -

def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

MODE = 'adaptive'
PARASITICS = True

# +
# Define various equations

if MODE == 'adaptive':
    neuron_eq = '''
        dVm/dt = ((glm + gpar) / Cm) * (Vm_r - Vm)  : volt
        dVt/dt = ((glt + gpar) / Ct) * (Vt_r - Vt)  : volt
        
        # dVm/dt = (glm / Cm) * (Vm_r - Vm) : volt
        # dVt/dt = (glt / Ct) * (Vt_r - Vt) : volt

        glm = flm * Cl                              : siemens
        glt = flt * Cl                              : siemens
        gpar = par_ctrl / par_leak_time * Cm : siemens
    '''
    reset_eq = '''
        Vm = Vm_r
        Vt = Vt * (Vt > Vm) + Vt_r * (Vt <= Vm)
    '''
    presyn_eq = '''
        Vm_old = Vm
        Vm = Vm_old + Vsyn
        Vt += (Cst/Ct) * (Vm_old - Vm_r)
    '''
else:
    neuron_eq = '''
        dVm/dt = (glm / Cm) * (Vm_r - Vm) : volt

        glm = flm * Cl                    : siemens
    '''
    reset_eq = '''
        Vm = Vm_r
    '''
    presyn_eq = '''
        Vm_old = Vm
        Vm = Vm_old + Vsyn
    '''

# Synapse equation is the same for both modes!
syn_eq = '''
    Vsyn = (W/Cm)*(Em - Vm) : volt
    Em                      : volt
    W                       : farad
'''

# +
# IFAT specific definitions
Vdd = 5 * volt
Cm = Ct = 0.44 * pF
Cl = 0.02 * pF

W_vals  = np.array([5, 10, 20, 40, 80]) * 0.001 * pF
Em_vals = np.array([0, 1/3, 2/3, 1]) * Vdd

par_ctrl = True
par_ctrl = float(PARASITICS)
par_leak_time = 12.5 * ms

# +
# Model parameters
Vm_r = 1.25 * volt
flm  = 0 * kHz
Csm  = W_vals[0]

Vt_r = 3 * volt
flt  = 0 * MHz
Cst  = 0 * pF

N = 4
# -

start_scope()

# Start stuff up, brochacho
test = NeuronGroup(N, neuron_eq, threshold='Vm > Vt', reset=reset_eq, method='exact')
test.Vm = Vm_r
test.Vt = Vt_r

start_spk_times = np.arange(0,0.05,0.002) * second
start_spk_inds  = np.zeros_like(start_spk_times)
#spgen = SpikeGeneratorGroup(1,in_spk_inds,in_spk_times)

in_spk_times = np.arange(0,1,0.0002) * second
in_spk_inds  = np.zeros_like(in_spk_times)
#spgen = PoissonGroup(1,rates=2*kHz)
spgen = SpikeGeneratorGroup(1,in_spk_inds,in_spk_times)

insyn = Synapses(spgen, test, syn_eq, on_pre=presyn_eq)
insyn.connect()
insyn.delay = "j * 0.2 * us"
insyn.Em = Em_vals[2]
insyn.W = W_vals[3]# + W_vals[2] + W_vals[0]

visualise_connectivity(insyn)

startup = SpikeGeneratorGroup(1,start_spk_inds, start_spk_times)
insyn2 = Synapses(startup, test, syn_eq, on_pre=presyn_eq)
insyn2.connect(i=0, j=0)
insyn2.Em = Em_vals[3]
insyn2.W = sum(W_vals)

# + {"scrolled": true}
exc_syn = Synapses(test, test, syn_eq, on_pre=presyn_eq)
exc_syn.connect('j==((i+1)%(4))')
exc_syn.connect('j==i')
exc_syn.Em = Em_vals[3]
exc_syn.W = sum(W_vals) + W_vals[1]
# -
inh_syn = Synapses(test, test, syn_eq, on_pre=presyn_eq)
inh_syn.connect('j==((i+2)%4)')
inh_syn.connect('j==((i+3)%4)')
inh_syn.Em = Em_vals[0]
inh_syn.W = sum(W_vals)*2

# +
#visualise_connectivity(inh_syn)

# +
#visualise_connectivity(exc_syn)
# -

sp_mon = SpikeMonitor(test)
vm_mon = StateMonitor(test, 'Vm', record=True)
vt_mon = StateMonitor(test, 'Vt', record=True)
ratemon = PopulationRateMonitor(test)

# + {"scrolled": true}
run(1*second)

# + {"scrolled": true}
for i in range(N):
    plot(vm_mon.t, vm_mon.Vm[i],vt_mon.t, vt_mon.Vt[i])
xlim([0,0.2])

# + {"scrolled": true}
plot(vm_mon.t, vm_mon.Vm[2]/volt,vt_mon.t, vt_mon.Vt[0]/volt); xlim([0,0.2]);
# -

plot(sp_mon.t, sp_mon.i, '.'); xlim([0,1]); ylim([-1, N+1])

plot(ratemon.t/ms, ratemon.smooth_rate(width=20*ms)/Hz)


