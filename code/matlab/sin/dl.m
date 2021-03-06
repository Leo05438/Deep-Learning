% layers=[ the number of neurons in each layer ]
layers=[1 2 1];
L=length(layers);

%{
           0 W2 W3 ... WL
network={                }
           0 b2 b3 ... bL
%}
for i=2:L
    network(1,i)={randn(layers(1,i),layers(1,i-1))};
    network(2,i)={randn(layers(1,i),1)};
end

% data
x=0:1/100:1;
y=sin(2*x*pi);
xn=length(x);


% Gradient Descent
eta=0.05;
times=1e5;

for count=1:times
    
    % Stochastic Gradient Descent
    %{
    %choose data
    k=randi(xn);
    input=x(:,k);
    output=y(:,k);
    
    %{
    forward pass
    a={ a1 a2 ... aL }
    %}
    a(1)={input};
    for i=2:L
        a(i)={activate(a{i-1},network{1,i},network{2,i})};
    end
    
    %{
    backward pass
    delta={ 0 delta2 delta3 ... deltaL }
    %}
    delta(L)={a{L}.*(1-a{L}).*(a{L}-output)};
    for i=2:(L-1)
        ni=L-i+1;
        delta(ni)={a{ni}.*(1-a{ni}).*(network{1,ni+1}'*delta{ni+1})};
    end
    
    % gradient step
    for i=2:L
        network{1,i}=network{1,i}-eta*delta{i}*a{i-1}';
        network{2,i}=network{2,i}-eta*delta{i};
    end
    %}
    
    % Gradient Descent
    %%{
    for k=1:xn
        input=x(:,k);
        output=y(:,k);

        %{
        forward pass
        a={ a1 a2 ... aL }
        %}
        a(1)={input};
        for i=2:L
            a(i)={activate(a{i-1},network{1,i},network{2,i})};
        end

        %{
         backward pass
         delta={ 0 delta2 delta3 ... deltaL }
        %}
        delta(L)={a{L}.*(1-a{L}).*(a{L}-output)};
        for i=2:(L-1)
            ni=L-i+1;
            delta(ni)={a{ni}.*(1-a{ni}).*(network{1,ni+1}'*delta{ni+1})};
        end

        % gradient step
        for i=2:L
            network{1,i}=network{1,i}-eta*delta{i}*a{i-1}'/10;
            network{2,i}=network{2,i}-eta*delta{i}/10;
        end
    end
    %}
    
    % compute cost
    newcost=cost(x,y,network)
    
end

N=500;
px=0:1/N:1;
for i=1:(N+1)
    input=px(i);
    %{
    forward pass
    a={ a1 a2 ... aL }
    %}
    a(1)={input};
        for j=2:L
            a(j)={activate(a{j-1},network{1,j},network{2,j})};
        end
    py(i)=a{L};
end

plot(px,py);
hold on;
plot(x,y);
hold off;











