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

# # ISCAS Paper Work
# See `ifat.py` for synaptic connetvitiy plots

# +
from brian2 import *
import matplotlib.pyplot as plt

# %matplotlib inline
# -

def visualize_connectivity(S):
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

def calc_weight(M, alpha, mu, sigma):
    output = zeros((M,M))
    for i in np.arange(M):
        for j in np.arange(M):
            output[i,j] = exp(cos((2*pi*i/M) - (2*pi*j/M) - mu)/sigma**2)
    output = output * (alpha/np.max(output))
    output = 5.0 * fF * np.around(output/(5.0*fF))
    return output

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
fF = 0.001 * pF
Vdd = 5 * volt
Cm = Ct = 440 * fF
Cl = 2 * fF

W_vals  = np.array([5, 10, 20, 40, 80]) * fF
Em_vals = np.array([0, 1/3, 2/3, 1]) * Vdd

par_ctrl = True
par_ctrl = float(PARASITICS)
par_leak_time = 12.5 * ms

# +
# Model parameters
Vm_r = 1 * volt
flm  = 0 * kHz
Csm  = W_vals[0]

Vt_r = 3.5 * volt
flt  = 0 * MHz
Cst  = 0 * fF

M = 64
# -

# Connectivity specifics
alpha = sum(W_vals)
mu1 = 0
mu2 = 2*pi/3
mu3 = pi
sigma = 36 * pi/180

# +
start_scope()

blair_exc = NeuronGroup(M, neuron_eq, threshold='Vm>Vt', reset=reset_eq, method='exact')
blair_exc.Vt = Vt_r
blair_exc.Vm = Vm_r

blair_inh = NeuronGroup(M, neuron_eq, threshold='Vm>Vt', reset=reset_eq, method='exact')
blair_inh.Vt = Vt_r
blair_inh.Vm = Vm_r

# +
exc2inh = Synapses(blair_exc, blair_inh, syn_eq, on_pre=presyn_eq)
exc2inh.connect()
exc2inh.Em = Em_vals[3]
exc2inh.W = calc_weight(M,alpha,mu1,sigma).flatten()

inh2exc = Synapses(blair_inh, blair_exc, syn_eq, on_pre=presyn_eq)
inh2exc.connect()
inh2exc.Em = Em_vals[0]
inh2exc.W  = calc_weight(M,alpha,mu2,sigma).flatten()

inh2inh = Synapses(blair_inh, blair_inh, syn_eq, on_pre=presyn_eq)
inh2inh.connect()
inh2inh.Em = Em_vals[0]
inh2inh.W = calc_weight(M,alpha,mu1,sigma).flatten()
# -

PoisIn = PoissonGroup(M,rates=2.0*kHz)
p2exc = Synapses(PoisIn, blair_exc, syn_eq, on_pre=presyn_eq)
p2exc.connect('j==i')
p2exc.Em = Em_vals[3]
p2exc.W = W_vals[2] + W_vals[0]# + W_vals[0]

i_spmon = SpikeMonitor(blair_inh)
e_spmon = SpikeMonitor(blair_exc)
e_vmon = StateMonitor(blair_exc, 'Vm', record=True)
erate0 = PopulationRateMonitor(blair_exc[:1])
erate1 = PopulationRateMonitor(blair_exc[1:2])
erate10 = PopulationRateMonitor(blair_exc[10:11])
erate20 = PopulationRateMonitor(blair_exc[13:14])
irate = PopulationRateMonitor(blair_inh[:1])

store()
poissonRates = arange(2.0,4.0,0.1)*kHz

print(poissonRates)

# + {"scrolled": true}
#rateOuts = zeros((4,20))
eRates0 = []
eRates1 = []
eRates10 = []
eRates20 = []
# -

for poissonRate in poissonRates:
    restore()
    PoisIn.rates = poissonRate
    #print(PoisIn.rates)
    run(8*second,report='text')
    eRates0.append(erate0.smooth_rate(width=1000*ms))
    eRates1.append(erate1.smooth_rate(width=1000*ms))
    eRates10.append(erate10.smooth_rate(width=1000*ms))
    eRates20.append(erate20.smooth_rate(width=1000*ms))

figure(figsize=(8,6))
plot(e_spmon.t/second, e_spmon.i,'.'); xlim([2,2.1])

rateWidth = 1000 * ms
plot(erate0.t/second,   erate0.smooth_rate(width=rateWidth)/Hz,
     erate1.t/second,   erate1.smooth_rate(width=rateWidth)/Hz,
     erate10.t/second, erate10.smooth_rate(width=rateWidth)/Hz,
     erate20.t/second, erate20.smooth_rate(width=rateWidth)/Hz)
xlim([2,6])

plot(irate.t/second,irate.smooth_rate(width=100*ms))

idx = 34
plot(e_vmon.t, e_vmon.Vm[idx]); xlim([0.2,0.8])
scatter(e_spmon.t[e_spmon.i==idx],3.65*ones(len(e_spmon.t[e_spmon.i==idx])),color='r')

shape(eRates0)

# +
smooth_width = 1000*ms

#avgRate = zeros_like(eRates0[0].smooth_rate(width=smooth_width))
avgRate = zeros((20,80000))

for i in range(20):
    avgRate[i,:]  =  eRates0[i]#.smooth_rate(width=smooth_width)
    avgRate[i,:] +=  eRates1[i]#.smooth_rate(width=smooth_width)
    avgRate[i,:] += eRates10[i]#.smooth_rate(width=smooth_width)
    avgRate[i,:] += eRates20[i]#.smooth_rate(width=smooth_width)
    avgRate[i,:] = avgRate[i,:]/4
# -

plot(erate0.t/second, avgRate[0,:]/Hz,
     erate0.t/second, avgRate[1,:]/Hz,
     erate0.t/second, avgRate[2,:]/Hz,
     erate0.t/second, avgRate[3,:]/Hz)

for i in range(20):    
    plot(erate0.t/second, avgRate[i,:]/Hz)
xlim([1.5,6.5])

shape(avgRate)

meanRates = np.mean(avgRate[:,15000:65000], axis=1)
stdRates  = np.std(avgRate[:,15000:65000], axis=1)

ax = gca()
ax.errorbar(poissonRates/1000, meanRates, yerr=stdRates)


