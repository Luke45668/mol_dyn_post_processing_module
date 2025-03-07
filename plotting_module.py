import numpy as np 
import matplotlib as plt
# generic functions for best fit lines
def quadfunc(x, a):
    return a * (x**2)


def linearfunc(x, a, b):
    return (a * x) + b


def linearthru0(x, a):
    return a * x


def powerlaw(x, a, n):
    return a * (x ** (n))


def plotting_stress_vs_strain(spring_force_positon_tensor_tuple,
                              i_1,i_2,j_,
                              strain_total,cut,aftcut,stress_component,label_stress,erate):
    

    mean_grad_l=[] 
    for i in range(i_1,i_2):
        #for j in range(j_):
            cutoff=int(np.round(cut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))
            aftcutoff=int(np.round(aftcut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))


            strain_plot=np.linspace(cut*strain_total,aftcut*strain_total,spring_force_positon_tensor_tuple[i][0,cutoff:aftcutoff,:,stress_component].shape[0])
            cutoff=int(np.round(cut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))
            aftcutoff=int(np.round(aftcut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))
            stress=np.mean(spring_force_positon_tensor_tuple[i][:,:,:,stress_component],axis=0)
            stress=stress[cutoff:aftcutoff]
            gradient_vec=np.gradient(np.mean(stress,axis=1))
            mean_grad=np.mean(gradient_vec)
            mean_grad_l.append(mean_grad)
            #print(stress.shape)
            # plt.plot(strain_plot,np.mean(stress,axis=1))
            # plt.ylabel(labels_stress[stress_component],rotation=0)
            # plt.xlabel("$\gamma$")
            # plt.plot(strain_plot,gradient_vec, label="$\\frac{dy}{dx}="+str(mean_grad)+"$")

            #plt.legend()
            #plt.show()

    plt.scatter(erate,mean_grad_l, label=label_stress)
    plt.xlabel("$\dot{\gamma}$")
    plt.ylabel("$\\frac{d\\bar{\sigma}_{\\alpha\\beta}}{dt}$", rotation=0,labelpad=20)
    #plt.show()

def plot_stress_tensor(t_0,t_1,
                    stress_tensor,
                    stress_tensor_std,
                    j_,n_plates, labels_stress,marker,cutoff,erate,e_end,ls_pick):
    for l in range(t_0,t_1):
            plt.errorbar(erate[cutoff:e_end], stress_tensor[cutoff:,l], yerr =stress_tensor_std[cutoff:,l]/np.sqrt(j_*n_plates), ls=ls_pick,label=labels_stress[l],marker=marker[l] )
            plt.xlabel("$\dot{\gamma}$")
            plt.ylabel("$\sigma_{\\alpha\\beta}$",rotation=0,labelpad=20)
    plt.legend()      
    #plt.show()