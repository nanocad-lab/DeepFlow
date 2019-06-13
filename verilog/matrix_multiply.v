module mat_mul(
    A,
    B,
    D,
    E,
    clk,
    reset
);

input clk, reset;
input [255:0] A;
input [255:0] B;
//input [511:0] C;
output wire [511:0] D;
output wire [31:0] E;

reg [15:0] A_int[3:0][3:0];
reg [15:0] B_int[0:3][0:3];
//reg [31:0] C_int[0:3][0:3];
reg [31:0] D_int[0:3][0:3];

assign D = {D_int[0][0],D_int[0][1],D_int[0][2],D_int[0][3],
            D_int[1][0],D_int[1][1],D_int[1][2],D_int[1][3],
            D_int[1][0],D_int[2][1],D_int[2][2],D_int[2][3],
            D_int[1][0],D_int[3][1],D_int[3][2],D_int[3][3]};
assign E = D_int[0][0];

integer i,j,k;

always@(posedge clk)
begin
    {A_int[0][0],A_int[0][1],A_int[0][2],A_int[0][3],
     A_int[1][0],A_int[1][1],A_int[1][2],A_int[1][3],
     A_int[2][0],A_int[2][1],A_int[2][2],A_int[2][3],
     A_int[3][0],A_int[3][1],A_int[3][2],A_int[3][3]} = A;

    {B_int[0][0],B_int[0][1],B_int[0][2],B_int[0][3],
     B_int[1][0],B_int[1][1],B_int[1][2],B_int[1][3],
     B_int[2][0],B_int[2][1],B_int[2][2],B_int[2][3],
     B_int[3][0],B_int[3][1],B_int[3][2],B_int[3][3]} = B;
     
/*    {C_int[0][0],C_int[0][1],C_int[0][2],C_int[0][3],
     C_int[1][0],C_int[1][1],C_int[1][2],C_int[1][3],
     C_int[2][0],C_int[2][1],C_int[2][2],C_int[2][3],
     C_int[3][0],C_int[3][1],C_int[3][2],C_int[3][3]} = C;*/
    if(reset)
    begin
        {D_int[0][0],D_int[0][1],D_int[0][2],D_int[0][3],
         D_int[1][0],D_int[1][1],D_int[1][2],D_int[1][3],
         D_int[1][0],D_int[2][1],D_int[2][2],D_int[2][3],
         D_int[1][0],D_int[3][1],D_int[3][2],D_int[3][3]} = 512'b0;
        i = 0;
        j = 0;
        k = 0;
    end
    else
    begin
       {D_int[0][0],D_int[0][1],D_int[0][2],D_int[0][3],
             D_int[1][0],D_int[1][1],D_int[1][2],D_int[1][3],
             D_int[1][0],D_int[2][1],D_int[2][2],D_int[2][3],
             D_int[1][0],D_int[3][1],D_int[3][2],D_int[3][3]} = 512'b0;
        for(i=0;i<4;i=i+1)
            for(j=0;j<4;j=j+1)
              //D_int[i][j] = C_int[i][j]; 
                for(k=0;k<4;k=k+1)
                    D_int[i][j] = D_int[i][j] + (B_int[i][k]*A_int[k][j]);
                              
    end
end


endmodule
