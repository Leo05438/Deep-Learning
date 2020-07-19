function costval=cost(x,y,network)

    xn=length(x);
    L=length(network);
    costvec=zeros(xn,1);
    
    for i=1:xn
        
        input=x(:,i);
        output=y(:,i);
        
        %{
        forward pass
        a={ a1 a2 ... aL }
        %}
        a(1)={input};
        for j=2:L
            a(j)={activate(a{j-1},network{1,j},network{2,j})};
        end
        
        costvec(i)=norm(output-a{L},2);
        
    end
    
    costval = norm(costvec,2)^2;