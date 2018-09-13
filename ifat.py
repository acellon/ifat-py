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

# +
from brian2 import *
import matplotlib.pyplot as plt

# %matplotlib inline
# -

MODE = 'adaptive'

# +
# Define various equations

if MODE == 'adaptive':
    neuron_eq = '''
        # dVm/dt = ((glm + parasitic) / Cm) * (Vm_r - Vm) : volt
        dVm/dt = (glm / Cm) * (Vm_r - Vm) : volt
        dVt/dt = (glt / Ct) * (Vt_r - Vt) : volt

        glm = flm * Cl                    : siemens
        glt = flt * Cl                    : siemens
        # parasitic = 1.0 / parasitic_leak_time * Cm
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

# +
# Model parameters
Vm_r = 1 * volt
flm  = 0 * kHz
Csm  = W_vals[0]

Vt_r = 3 * volt
flt  = 0.2 * MHz
Cst  = 0 * pF

N = 4
# -

start_scope()

# Start stuff up, brochacho
test = NeuronGroup(N, neuron_eq, threshold='Vm > Vt', reset=reset_eq, method='exact')
test.Vm = Vm_r
test.Vt = Vt_r

in_spk_times = np.arange(0,1,0.01) * second
in_spk_inds  = np.zeros_like(in_spk_times)
spgen = SpikeGeneratorGroup(1,in_spk_inds,in_spk_times)

insyn = Synapses(spgen, test, syn_eq, on_pre=presyn_eq)
insyn.connect()
insyn.delay = "j * 0.2 * us"
insyn.Em = Em_vals[-1]
insyn.W = sum(W_vals)

# +
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

visualise_connectivity(insyn)


# +
# insyn = Synapses(spgen, test, syn_eq, on_pre=presyn_eq, multisynaptic_index='k')
# insyn.connect(i=0,j=0,n=100)
# insyn.delay='0.2*us+k*0.2*us'
# insyn.Em = Em_vals[1]
# insyn.W = Csm

# +
# exc_syn = Synapses(test, test, syn_eq, on_pre=presyn_eq)
# exc_syn.connect('j==((i+1)%6)')
# exc_syn.Em = '4*volt'
# exc_syn.W = 30 * Csm
# -
sp_mon = SpikeMonitor(test)
vm_mon = StateMonitor(test, 'Vm', record=True)
vt_mon = StateMonitor(test, 'Vt', record=True)

# + {"scrolled": true}
run(1*second)

# + {"scrolled": true}
for i in range(N):
    plot(vm_mon.t, vm_mon.Vm[i],vt_mon.t, vt_mon.Vt[i])

# + {"scrolled": true}
plot(vm_mon.t, vm_mon.Vm[0],vt_mon.t, vt_mon.Vt[0]); xlim([0,0.01])

# + {"scrolled": true}
for i in range(6):
    plot(vmon.t, vmon.Vmem[i])
    
xlim([0.098,0.102])
# -

plot(mon.t, mon.i,'.'); xlim([0.099,0.11])

insyn

insyn.delay


