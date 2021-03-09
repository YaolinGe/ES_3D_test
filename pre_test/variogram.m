      clc;
     close all;
     clear all;
     x=10+sqrt(50)*randn(400,1);
     y=2+sqrt(30)*randn(400,1);
     [a,b]=size(x);
     d=a/40;           %lag distance separation
     num=a/d;
     
     for i=1:num
     l=(x(1)-x(d))^2+ (y(1)-y(d))^2+ (y(1)-y(d))^2;
     lam(i)=(0.5*l)/3;              %varigram calculation
     di(i)=d;
     d=d+10;
     end
       marker = 'o--';
       plot(di,lam,marker);
        axis([0 400 0 max(lam)*1.1]);
        xlabel('h');
        ylabel('\gamma (h)');
        title('(Semi-)Variogram');
        
        
        
%% test of variogram on my own




