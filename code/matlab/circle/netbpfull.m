function netbp_full
%NETBP_FULL
%   Extended version of netbp, with more graphics
%
%   Set up data for neural net test
%   Use backpropagation to train 
%   Visualize results
%
% C F Higham and D J Higham, Aug 2017
%
%%%%%%% DATA %%%%%%%%%%%
% xcoords, ycoords, targets
%{
x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
y = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];
%}
xn=100;
x1=rand(1,xn);
x2=rand(1,xn);
y=zeros(2,xn);
for j=1:xn
    x1v=x1(1,j);
    x2v=x2(1,j);
    if ((x1v-0.5)*(x1v-0.5)+(x2v-0.5)*(x2v-0.5)<=0.25)
        y(1,j)=1;
        y(2,j)=0;
    else
        y(1,j)=0;
        y(2,j)=1;
    end
end

figure(1)
clf
a1 = subplot(1,1,1);
%{
plot(x1(1:xn),x2(1:xn),'ro','MarkerSize',1,'LineWidth',4)
hold on
plot(x1(6:10),x2(6:10),'bx','MarkerSize',12,'LineWidth',4)
%}
for k=1:xn
    if (y(1,k)==1)
        plot(x1(k),x2(k),'ro','MarkerSize',1,'LineWidth',4)
        if k~=xn
            hold on
        end
    else
        plot(x1(k),x2(k),'bx','MarkerSize',1,'LineWidth',4)
        if k~=xn
            hold on
        end
    end
end
a1.XTick = [0 1];
a1.YTick = [0 1];
a1.FontWeight = 'Bold';
a1.FontSize = 16;
xlim([0,1])
ylim([0,1])

%print -dpng pic_xy.png

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize weights and biases 
%rng(5000);
%rng('shuffle');
W2 = 0.5*randn(2,2);
W3 = 0.5*randn(3,2);
W4 = 0.5*randn(2,3);
b2 = 0.5*randn(2,1);
b3 = 0.5*randn(3,1);
b4 = 0.5*randn(2,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forward and Back propagate 
% Pick a training point at random
eta = 0.025;
Niter = 1e6;
savecost = zeros(Niter,1);
for counter = 1:Niter
    k = randi(10);
    x = [x1(k); x2(k)];
    % Forward pass
    a2 = activate(x,W2,b2);
    a3 = activate(a2,W3,b3);
    a4 = activate(a3,W4,b4);
    % Backward pass
    delta4 = a4.*(1-a4).*(a4-y(:,k));
    delta3 = a3.*(1-a3).*(W4'*delta4);
    delta2 = a2.*(1-a2).*(W3'*delta3);
    % Gradient step
    W2 = W2 - eta*delta2*x';
    W3 = W3 - eta*delta3*a2';
    W4 = W4 - eta*delta4*a3';
    b2 = b2 - eta*delta2;
    b3 = b3 - eta*delta3;
    b4 = b4 - eta*delta4;
    % Monitor progress
    newcost = cost(W2,W3,W4,b2,b3,b4)   % display cost to screen
    savecost(counter) = newcost;
end

figure(2)
clf
semilogy([1:Niter/500:Niter],savecost(1:Niter/500:Niter),'b-','LineWidth',2)
xlabel('Iteration Number')
ylabel('Value of cost function')
set(gca,'FontWeight','Bold','FontSize',18)
print -dpng pic_cost.png

%%%%%%%%%%% Display shaded and unshaded regions 
N = 500;
Dx = 1/N;
Dy = 1/N;
xvals = [0:Dx:1];
yvals = [0:Dy:1];
for k1 = 1:N+1
    xk = xvals(k1);
    for k2 = 1:N+1
        yk = yvals(k2);
        xy = [xk;yk];
        a2 = activate(xy,W2,b2);
        a3 = activate(a2,W3,b3);
        a4 = activate(a3,W4,b4);
        Aval(k2,k1) = a4(1);
        Bval(k2,k1) = a4(2);
     end
end
[X,Y] = meshgrid(xvals,yvals);

figure(3)
clf
a2 = subplot(1,1,1);
Mval = Aval>Bval;
contourf(X,Y,Mval,[0.5 0.5])
hold on
colormap([1 1 1; 0.8 0.8 0.8])
%{
plot(x1(1:5),x2(1:5),'ro','MarkerSize',12,'LineWidth',4)
plot(x1(6:10),x2(6:10),'bx','MarkerSize',12,'LineWidth',4)
%}
for k=1:xn
    if (y(1,k)==1)
        plot(x1(k),x2(k),'ro','MarkerSize',1,'LineWidth',4)
        if k~=xn
            hold on
        end
    else
        plot(x1(k),x2(k),'bx','MarkerSize',1,'LineWidth',4)
        if k~=xn
            hold on
        end
    end
end
a2.XTick = [0 1];
a2.YTick = [0 1];
a2.FontWeight = 'Bold';
a2.FontSize = 16;
xlim([0,1])
ylim([0,1])

print -dpng pic_bdy_bp.png

  function costval = cost(W2,W3,W4,b2,b3,b4)

     costvec = zeros(xn,1); 
     for i = 1:xn
         x =[x1(i);x2(i)];
         a2 = activate(x,W2,b2);
         a3 = activate(a2,W3,b3);
         a4 = activate(a3,W4,b4);
         costvec(i) = norm(y(:,i) - a4,2);
     end
     costval = norm(costvec,2)^2;
   end % of nested function

end
