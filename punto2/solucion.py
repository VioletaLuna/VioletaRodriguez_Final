import numpy as np
import matplotlib.pyplot as plt

m = 1.0
sigma=1.0

data = np.genfromtxt("datos_observacionales.dat", delimiter = " ", skip_header = 0)
x_obs = data[:,0]
y_obs= data[:,1]
z_obs = data[:,2]


#La función de densidad a samplear es el posteriro dado por bayes.

#Resolvemos por diferencia finitas. Parametros: sigma, rho, beta
def modelo(x,y,z, param):

    delta = 1E-5
    x1= x + param[0]*(y-x)*delta
    y1=y +x*(param[1]-z)-y
    z1= z + x*y -param[2]*z
    
    return x, y,z
   	
   	
	
def loglikelihood(x_obs, y_obs,z_obs, param):
		x,y,z =modelo(x_obs, y_obs,z_obs, param)
		temx = x_obs-x
		temy = y_obs-y
		temz = z_obs-z 
				 
		temx=temy/sigma
		temx=temy**2
		temx= np.sum(temy)*(-0.5)

		temy=temy/sigma
		temy=temy**2
		temy= np.sum(temy)*(-0.5)
			 
		temz=temz/sigma
		temz=temz**2
		temz= np.sum(temz)*(-0.5)
		return temx + temz + temy

#Lo más conveniente es considera un prior gausiano
def logprior(param):
    tem=param**2/(2*(sigma)**2)
    tem = -0.5 * np.sum(tem)
    return tem  

#Hacemos la derivada de forma númerica.
def der_loglikelihood(x_obs, y_obs,z_obs, param):
    delta= 1E-5
    der=np.zeros(param.size)
    for i in range(param.size):
        param[i]+= delta
        adelante = loglikelihood(x_obs, y_obs,z_obs, param)
        param[i]= param[i] - 2*delta
        atras= loglikelihood(x_obs, y_obs,z_obs, param)
        der[i] = (adelante-atras)/(2.0 * delta)
        param[i]+= delta
    return der

#Tomamos un prior gaussiano
def der_logprior(param):
    tem = -0.5*param/sigma**2
    return tem
    

#Es la derivada con rspecto a los parametros. 
def derivada_Hq(x_obs, y_obs,z_obs, param):
    tem = -der_loglikelihood(x_obs, y_obs,z_obs, param) #-der_logprior(param) 
    return tem

def derivada_Hp(p,m):
    return p/m

def H(x_obs, y_obs,z_obs, p, param):
    p= np.sum(p*p)
    U= p*p/(2*m)
    K=-loglikelihood(x_obs, y_obs,z_obs, param)#-logprior(param)
   
    return U+K

#Resolvemos por medio de leapfrog el sistema de ecuaciones de hamilton. Usamos kick - drift- kick.
#Además, hago varias veces un paso para no avanzar en pasos tan cortos.
def leapfrog(x_obs, y_obs,z_obs, p , param, delta=1E-1, iteraciones=2):
    paramNew =param.copy()
    pNew=p
   
    
    for i in range(iteraciones):
        #kick
        pNew = pNew - derivada_Hq(x_obs, y_obs,z_obs, paramNew)*delta*0.5
        #Drift
        paramNew = paramNew + derivada_Hp(pNew, m)*delta
        #kick
        pNew = pNew - derivada_Hq(x_obs, y_obs,z_obs, paramNew)*delta*0.5
    return paramNew, pNew

#Resolvemos por medio del método: Marcov chain - Monte Carlo. 
def MCMC(pasos):
    param = np.zeros((pasos,3))
    p = np.zeros((pasos,3))
    
    #Establezco las condiciones iniciales
    #Hay un p para cada parametro, el momentum también es en varias dimesiones
    p[0,:] = np.random.normal(size=3, scale=0.3)
    #Por ahora, definamos así la condición incial de los paremetros, de ser necesaraio luego lo cambiamos.
    param[0,:] = np.random.normal(size=3, scale=0.3)
    
    for i in range(1,pasos):
        
        prop_param, prop_p= leapfrog(x_obs, y_obs,z_obs, p[i-1,:], param[i-1,:])
        prop_p = -prop_p #negamos a p para que la propuesta sea simetrica.
        
        alpha=np.random.random()
        r = min(1, np.exp(-(H(x_obs, y_obs,z_obs,prop_p ,prop_param)-H(x_obs, y_obs,z_obs, p[i-1,:], param[i-1,:]))))
				
        if(alpha<r):
            param[i,:]=prop_param
            p[i,:]=prop_p
        else:
            param[i,:]=param[i-1,:]
            p[i,:]=p[i-1,:]
            
    return param 

parametros=MCMC(100)

plt.hist(paramtros[0], label="Sigma")
np.mean(paramtros[0])

plt.hist(paramtros[1], label="Rho")
np.mean(paramtros[2])

plt.hist(paramtros[2], label="Beta")
np.mean(paramtros[2])

plt.label()
plt.savefig("parametros.png")


