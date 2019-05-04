import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Parameters
h = 1.
g = 0.1
m = 10.
Fmax = 10.
# p0 = np.matrix('50 ;50; 100')
# v0 = np.matrix('-10; 0; -10')
p0 = np.array([50,50,100])
v0 = np.array([-10,0,-10])
alpha = 0.5
gamma = 1.
K = 35

gv = np.array([0,0,g])

# CVXPY problem setup
p = cp.Variable((3,K)) # position vectors
v = cp.Variable((3,K)) # velocity vectors
f = cp.Variable((3,K)) # thrust vectors

fuel = 0
for k in range(K):
    fuel = fuel + gamma * h * cp.norm(f[:,k], 2)

objective = cp.Minimize(fuel)
constraints = [
    # Initial state
    p[:,0] == p0,
    v[:,0] == v0,
    # Target
    p[:,-1] == 0,
    v[:,-1] == 0,
]

for k in range(K):
    constraints += [
        # Maximal thrust
        cp.norm(f[:,k], 2) <= Fmax,
        # Glide cone. The spacecraft must remain in this region
        p[2,k] >= alpha * cp.norm(p[:2,k], 2)
    ]

# Spacecraft dynamics constraints
for k in range(K-1):
    constraints += [
        v[:,k+1] == v[:,k] + h/m*f[:,k] - h*gv,
        p[:,k+1] == p[:,k] + h/2*(v[:,k] + v[:,k+1])
    ]

prob = cp.Problem(objective, constraints)
res = prob.solve(verbose=True)

print(res)
print(prob.status)


# use the following code to plot your trajectories
# and the glide cone (don't modify)
# -------------------------------------------------------
fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.linspace(-40, 55, num=30)
Y = np.linspace(0, 55, num=30)
X, Y = np.meshgrid(X, Y)
Z = alpha*np.sqrt(X**2+Y**2)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
#Have your solution be stored in p
ax.plot(xs=p.value[0,:],ys=p.value[1,:],zs=p.value[2,:])
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
plt.show()

plt.plot(p.value.T)
plt.show()


# Use the following code to print the inactive variables
# ------------------------------------------------------
def analyze_constraints(constraints, max_k):
    '''
    Prints the inactive constraints and returns a heatmap of constraint activities
    '''
    k = 0
    
    
    constraint_activity = np.ones((8, max_k+1))
    
    #check values for each constraints
    for cons in constraints:
        #if constraint is scalar (from k = 4 to 2max_k+3)
        if cons.shape != (3,):   
            #if value too low
            if cons.dual_value < 1e-5:
                if k%2 == 0: #even scalar constraints are for max thrust constraints
                    step = (k-2)/2 - 1
                    print('Max thrust constraint inactive at step k =', step)  
                    constraint_activity[0, int(step)] = 0  
                else: #odd scalar constraints are for glide cone constraints
                    step = (k-1)/2 - 2
                    print("Glide cone constraint inactive at step k =", step)
                    constraint_activity[1, int(step)] = 0
        #if constraint is an arry (from k = 0 to 4 and from 2max_k+4 to 4max_k+1)
        else:
            test = np.where(np.abs(cons.dual_value) < 0.2)
            if len(test[0])> 0: #if values are lower than 1
                if k%2 == 0: #even array constraints are speed dynamic constraints
                    situation = "Speed dynamic constraints"
                    step = (k - (2*max_k))/2
                    for elem in test[0]:
                        print(situation, 'constraint inactive at step k = ', step)
                        constraint_activity[2+elem, int(step)] = 0
                else: #odd array constraints are thrust dynamic constraints
                    situation = "Thrust dynamic constraints"
                    step = (k - (2*max_k + 1))/2
                    for elem in test[0]:
                        print(situation, 'constraint inactive at step k = ', step)
                        constraint_activity[5+elem, int(step)] = 0
        k = k+1
        
    return constraint_activity

constraint_activity = analyze_constraints(prob.constraints, K)

# plot constrain heatmap:
fig, ax = plt.subplots(figsize=(10,5)) 
yticks = np.array(['Max Thrust Constraint', 'Glide Cone Constraint', 
                   'Speed Constraint $x$', 'Speed Constraint $y$', 'Speed Constraint $z$', 
                  'Thrust Constraint $x$', 'Thrust Constraint $y$', 'Thrust Constraint $z$'])
ax = sns.heatmap(constraint_activity, yticklabels=yticks, cmap='Blues') 

ax.set_ylabel('Constraint')
ax.set_xlabel('Time Step')
ax.set_title('Constraint Activity Over Time')
plt.show()
