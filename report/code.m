h = 1;
g = 0.1;
m = 10;
Fmax = 10;
p0 = [50,50,100]';
v0 = [-10,0,-10]';
alpha = 0.5;
gamma = 1;
K = 35;

cvx_begin
    % position, velocity, thrust vectors
    variables p(3,K) v(3,K) f(3,K);

    fuel = 0;
    for k = 1:K
        fuel = fuel + gamma * h * norm(f(:,k),2);
    end

    minimize ( fuel );
    subject to
        % Initial state
        p(:,1) == p0;
        v(:,1) == v0;
        % Target
        p(:,K) == zeros(3,1);
        v(:,K) == zeros(3,1);
        for k = 1:K
            % Maximal thrust
            norm(f(:,k), 2) <= Fmax;
            % Glide cone. The spacecraft must remain in this region
            p(3,k) >= alpha * norm(p(1:2,k), 2)
        end
        % Spacecraft dynamics constraints
        for k = 1:K-1
            v(:,k+1) == v(:,k) + h/m*f(:,k) - h*[0,0,g]';
            p(:,k+1) == p(:,k) + h/2*(v(:,k) + v(:,k+1));
        end
cvx_end

x = linspace(-40,55,30); y = linspace(0,55,30);
[X,Y] = meshgrid(x,y);
Z = alpha*sqrt(X.^2+Y.^2);
figure; colormap autumn; surf(X,Y,Z);
axis([-40,55,0,55,0,105]);
grid on; hold on;
plot3(p(1,:),p(2,:),p(3,:),'b','linewidth',1.5);
quiver3(p(1,1:K),p(2,1:K),p(3,1:K),...
        f(1,:),f(2,:),f(3,:),0.3,'k','linewidth',1.5);

