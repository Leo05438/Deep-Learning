function y = activate(x,W,b)
% activate function
y=1./(1+exp(-(W*x+b)));
