import torch

def CalContrast(Input):
   Input=Input.cuda()
   h, w, c,x = Input.shape
   Contrast = (torch.zeros(1, w, 1, 1)).cuda()
   #for i in range(w):
      # Contrast[0,i,0,0]=Input[0,i,:,:].std();
   Contrast[0,:,0,0]=torch.std(Input, dim=[2,3])
     #''' min = abs(torch.min(Input[0,i,:,:]))
     # max = torch.max(Input[0,i,:,:])
     # if (max+min)==0:
      #   Contrast[0,i,0,0]=max
     # else:
     #    Contrast[0,i,0,0]= (abs(max-min)/(max+min)'''
      #print('Hellow')  
        
      #Contrast[0,i,0,0]=1
     # Contrast=Contrast.cuda()
   return(Contrast) 

