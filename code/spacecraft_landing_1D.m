h = 1;
g = 0.1;
m = 10;
Fmax = 10;
p0 = 100;
v0 = -10;
alpha = 0.5;
gamma = 1;
N = 35;

cvx_begin
    variables p(N) v(N) f(N);
    minimize ( norm(f,1) );
    subject to
        p(N) == 0;
        v(N) == 0;
        p(1) == p0;
        v(1) == v0;
        for k = 1:N-1
            v(k+1) == v(k) + h/m*f(k) - h*g;
            p(k+1) == p(k) + h/2*(v(k) + v(k+1));
        end
cvx_end

subplot(3,1,1);
plot(p);
subplot(3,1,2);
plot(v);
subplot(3,1,3);
plot(f);
