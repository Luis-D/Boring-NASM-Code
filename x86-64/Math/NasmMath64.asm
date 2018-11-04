; To be used with NASM-Compatible Assemblers

; System V AMD64 ABI Convention (*nix)
; Function parameters are passed this way:
; Interger values: RDI, RSI, RDX, RCX, R8, R9
; Float Point values: XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7
; Extra arguments are pushed on the stack starting from the most left argument
; Function return value is returned this way:
; Integer value: RAX:RDX 
; Float Point value: XMM0:XMM1
; RAX, R10 and R11 are volatile

; Microsoft x64 calling convention (Windows)
; Function parameters are passed this way:
; Interger values: RCX, RDX, R8, R9
; Float Point values: XMM0, XMM1, XMM2, XMM3 
; Both kind of arguments are counted together
; e.g. if the second argument is a float, it will be in arg2f, if Interger, then RDX
; Extra arguments are pushed on the stack 
; Function return value is returned this way:
; Integer value: RAX
; Float Point value: XMM0
; XMM4, XMM5, RAX, R10 and R11 are volatile

; SSE, SSE2, SSE3

; Luis Delgado.

;October 4, 2018:   Fixing Windows support, weird behavior @ V2ScalarMUL
;October 2, 2018:   Adding Windows support.
;November 26, 2018: Date of last edition (before translation).

; Collection of mathematical functions, very useful for algebra.



;*****************************
;MACROS
;*****************************

%ifidn __OUTPUT_FORMAT__, elf64 
%elifidn __OUTPUT_FORMAT__, win64
%endif

%ifidn __OUTPUT_FORMAT__, win64 
%endif

%macro args_reset 0
    %ifidn __OUTPUT_FORMAT__, elf64 
        %define arg1 RDI
        %define arg2 RSI
        %define arg3 RDX
        %define arg4 RCX
        %define arg5 R8
        %define arg6 R9 
    %elifidn __OUTPUT_FORMAT__, win64
        %define arg1 RCX
        %define arg2 RDX
        %define arg3 R8
        %define arg4 R9
        %define arg5 [rbp+48]
        %define arg6 [rbp+48+8]
    %endif

    %define arg1f XMM0
    %define arg2f XMM1
    %define arg3f XMM2
    %define arg4f XMM3
%endmacro


args_reset ;<--Sets arguments definitions to normal, as it's definitions can change.

%macro TRANS44 0
    movaps  xmm4,  arg1f
    movaps  xmm6,  arg3f

    punpckldq arg1f,arg2f
    punpckldq arg3f,arg4f
    punpckhdq xmm4,arg2f
    punpckhdq xmm6,arg4f

    movaps  arg2f,arg1f
    movaps  xmm5,xmm4

    punpcklqdq arg1f,arg3f
    punpckhqdq arg2f,arg3f
    punpcklqdq xmm4,xmm6
    punpckhqdq xmm5,xmm6
%endmacro

%macro CROSSPRODUCTMACRO 6
;%1 First register to use
;%2 Second register to use
;%3 Register to store result
;%4, %5, %6 temporal registers
       movups %3,%1
        ;%3 [?][Az][Ay][Ax]
        movups %4,%2
        ;%4 [?][Bz][By][Bx]
        pshufd %5,%3,11010010b
        pshufd %6,%4,11001001b
        pshufd %3,%3,11001001b
        pshufd %4,%4,11010010b
        ;%3 [?][Bx][Bz][By]
        ;%4 [?][By][Bx][Bz]
        ;%5 [?][Ay][Ax][Az]
        ;%6 [?][Bx][Bz][By]
        mulps %3,%4
        mulps %5,%6
        subps %3,%5
        ;%3 [?][Rz][Ry][Rx]
%endmacro

%macro NORMALIZEVEC3MACRO 4 
;%1 Register with the vector about to normalize
;%2, %3 y %4 temporal registers

    movaps %3,%1
    ;%3 [][z][y][x]
    mulps  %3,%1
    ;%3 [][z*z][y*y][x*x]	
	movhlps %2,%3
	;%2 [][][][z*z]
	pshufd %4,%3,1
	;%4 [][][][y*y]
	addss %3,%4
	addss %3,%2
	;%3 [][][][(x*x)+(y*y)+(z*z)]
	sqrtss %3,%3
	;%3 [][][][sqrt((x*x)+(y*y)+(z*z))]
    pshufd %3,%3,0
    divps %1,%3
%endmacro 

%macro MULVEC4VEC4 3
        movups arg3f,[%1+%3]
        movaps xmm7,arg3f

        mulps  arg3f,arg1f
        movshdup    arg4f, arg3f
        addps       arg3f, arg4f
        movaps xmm6,xmm7
        movhlps     arg4f, arg3f
        addss       arg3f, arg4f
        movss  [%2+%3], arg3f;//<--- Important

        mulps  xmm6,arg2f
        movshdup    arg4f, xmm6
        addps       xmm6, arg4f
        movaps arg3f,xmm7
        movhlps     arg4f, xmm6
        addss       xmm6, arg4f
        movss  [%2+4+%3], xmm6;//<--- Important

        mulps  arg3f,xmm4
        movshdup    arg4f, arg3f
        addps       arg3f, arg4f
        movaps xmm6,xmm7
        movhlps     arg4f, arg3f
        addss       arg3f, arg4f
        movss  [%2+8+%3], arg3f;<--- Important

        mulps  xmm6,xmm5
        movshdup    arg4f, xmm6
        addps       xmm6, arg4f
        movhlps     arg4f, xmm6
        addss       xmm6, arg4f
        movss  [%2+8+4+%3], xmm6;<--- Important

%endmacro

%macro DotProductXMM 4
;%1 and %2 are registers to proccess
;%3 is the result ;Result stored in the first 32-bits
;%4 is a temporal register
	movaps %3, %1
	mulps  %3, %2
	movshdup %4,%3
	addps    %3,%4
	movhlps  %4,%3
	addss    %3,%4
%endmacro

%macro DotProductXMMV3 4
;%1 and %2 are registers to proccess
;%3 is the result ;Result stored in the first 32-bits
;%4 is a temporal register
	movaps %3, %1
	;%3[?][z1][y1][z1]
	mulps  %3, %2
	;%3[?][z1*z2][y1*y2][x1*x2]
	movshdup %4,%3
	;%4[?][?]    [y1*y2][y1*y2]
	addps    %4,%3
	;%4[?][?]    [?]    [(x1*x2)+(y1*y2)]
	movhlps  %3,%3
	;%3[?][z1*z2][?]    [z1*z2]
	addss    %3,%4
	;%3[?][?]    [?]    [(x1*x2)+(y1*y2)+(z1*z2)]
%endmacro

%macro M4x4MULMACRO 2 ;When used, registers should be 0'd
    DotProductXMM arg4f,%1,xmm8,xmm9
    pshufd xmm10,xmm8,0
    DotProductXMM arg3f,%1,xmm8,xmm9
    movss xmm10,xmm8
    DotProductXMM arg2f,%1,xmm8,xmm9
    pshufd %2,xmm8,0
    DotProductXMM arg1f,%1,xmm8,xmm9
    movss %2,xmm8
    movlhps %2,xmm10
%endmacro


;********************************************************************
; .data
;********************************************************************

section .data

  ;***** Constants *****;
    fc_360f:                   equ 0x43b40000                         ;32-bits 360.f
    fc_2f:                     equ 01000000000000000000000000000000b  ;32-bits 2.f
    fc_m_1f:                   equ 0xbf800000                         ;32-bits -1.f
    fc_1f:                     equ 0x3f800000                         ;32-bits +1.f
    SignChange32bits:          equ 10000000000000000000000000000000b  ;It can change the sign if XOR'd
    fc_180f:                   equ 0x43340000                         ;32-bits 180.f
    fc_PIdiv180f:              equ 0x3c8efa35                         ;32-bits (PI/180.f)
    fc_180fdivPI:              equ 0x42652ee1                         ;32-bits (180.f/PI)

;***** Variables *****;
    fc_PIdiv180f_mem:           dd 0x3c8efa35                     ;32-bits (PI/180.f)




;********************************
; CODE
;********************************

section .text


global V2Rotate_FPU; 
;void V2Rotate_FPU(float * Vec2_A, float * Angle_Degrees, float * Vec2_Result);
;****************************************************************
;It rotates 2D point A around (0,0) by and angle given in degrees.
;****************************************************************
V2Rotate_FPU:
    enter 0,0

    sub rsp,8

    fld dword [fc_PIdiv180f_mem]
        pxor xmm4,xmm4
        pxor arg4f,arg4f
        
    fmul dword [arg2]
    pcmpeqd xmm4,xmm4
        ;xmm4[0] = [11111111111111111111111111111111]
        
    fld st0
    fcos 
    fstp dword [rsp]
       pslld xmm4,31
            ;xmm4[0] = [10000000000000000000000000000000]
          movss arg4f,xmm4
    fsin
    fstp dword [rsp+4]

    movsd arg1f,[rsp]
    ;arg1f [][][sin][cos]
    pshufd arg2f,arg1f,11_10_00_01b
    ;arg2f [][][cos][sin]
    movsd arg3f,[arg1]
    movsldup xmm5,arg3f
    ;xmm5 [][][x][x]
    movshdup xmm6,arg3f
    ;xmm6 [][][y][y]
    pxor xmm6,arg4f
    ;xmm6 [][][y][-y]

    mulps arg1f,xmm5
    mulps arg2f,xmm6
    addps arg1f,arg2f

    movsd [arg3],arg1f

    add rsp,8
    leave 
    ret

global V2ScalarMUL;V2ScalarMUL(float * Vec2_A, float Scalar_B, float * Vec2_Result);
;**********************************************
;Given A (2D Vectors) and B (an Scalar value),
;2D Vector Result = (Ax * B , Ay * B);
;**********************************************
V2ScalarMUL:

%ifidn __OUTPUT_FORMAT__, win64 
    %define arg2 R8
    %define arg1f XMM0 ;<- I don't know why it works, it should be XMM1, isn't it?
%endif
    enter 0,0
    movsd arg2f, [arg1]
    movsldup arg1f,arg1f
    mulps arg1f,arg2f
    movsd [arg2],arg1f
    leave
    ret 
%ifidn __OUTPUT_FORMAT__, win64 
args_reset
%endif



global V2V2ADD; V2V2ADD(float * Vec2_A, float * Vec2_B, float * Vec2_Result);
;**********************************************
;Given A and B, 2D Vectors,
;2D Vector Result = (Ax + Bx , Ay + By);
;**********************************************
V2V2ADD:
    enter 0,0
    movsd arg1f, [arg1]
    movsd arg2f, [arg2]
    addps arg1f,arg2f
    movsd [arg3],arg1f
    leave
    ret

global V2V2SUB; V2V2SUB(float * Vec2_A, float * Vec2_B, float * Vec2_Result);
;**********************************************
;Given A and B, 2D Vectors,
;2D Vector Result = (Ax - Bx , Ay - By);
;**********************************************
V2V2SUB:
    enter 0,0
    movsd arg1f, [arg1]
    movsd arg2f, [arg2]
    subps arg1f,arg2f
    movsd [arg3],arg1f
    leave
    ret


global V3V3ADD; V3V3ADD(float* A, float* B, float* Resultado);
;*********************************************************
;Given A and B (both 3D vectors),
;3D Vector Result = (Ax + Bx, Ay + By, Az + Bz, Aw + Bw);
;*********************************************************
V3V3ADD:
    enter 0,0
	movups arg1f,[arg1]
        movups arg2f,[arg2]
        addps arg1f,arg2f
	movhlps arg2f,arg1f
        movsd [arg3],arg1f
	movss [arg3+8],arg2f
    leave
    ret

global V3V3SUB;  void V3V3SUB(float * Vec3_A, float * Vec3_B,float * Vec3_Result);
;**********************************************
;Given A and B, 2D Vectors,
;2D Vector Result = (Ax - Bx , Ay - By);
;**********************************************
V3V3SUB:
    enter 0,0
    movsd arg1f, [arg1]
    movsd arg2f, [arg2]
    subps arg1f,arg2f

    movss arg2f,[arg1+8]
    movss arg3f,[arg2+8]
    subss arg2f,arg3f

    movsd [arg3],arg1f
    movss [arg3+8],arg2f


    leave
    ret


global V4V4ADD; void V4V4ADD(float * A, float *B, float * Result);
;*********************************************************
;Given A and B (both 4D vectors),
;4D Vector Result = (Ax + Bx, Ay + By, Az + Bz, Aw + Bw);
;*********************************************************
V4V4ADD:
    enter 0,0
        movups arg1f,[arg1]
        movups arg2f,[arg2]
        addps arg1f,arg2f
        movups [arg3],arg1f
    leave
    ret
    
global V4V4SUB; void V4V4ADD(float * A, float *B, float * Result);
;*********************************************************
;Given A and B (both 4D vectors),
;4D Vector Result = (Ax - Bx, Ay - By, Az - Bz, Aw - Bw);
;*********************************************************
V4V4SUB:
    enter 0,0
        movups arg1f,[arg1]
        movups arg2f,[arg2]
        subps arg1f,arg2f
        movups [arg3],arg1f
    leave
    ret


global V2CalculatePerpendicular
; V2CalculatePerpendicular(float * Vec2_A,float * Vec2_Result,float ClockWise_Multiplier)
;**********************************************
;Given a 2D Vector A,
;This function calculates a perpendicular vector. Direction depends on the third parameter:
;ClockWise_Multiplier should be 1.f or -1.f
;**********************************************
V2CalculatePerpendicular:

%ifidn __OUTPUT_FORMAT__, win64 
    %define arg1f XMM2
%endif
    enter 0,0

    movsd arg2f, [arg1]
    movsldup arg1f,arg1f
    
    pxor xmm4,xmm4
    pcmpeqd xmm4,xmm4
    ;xmm4[0] = [11111111111111111111111111111111]
    pxor arg4f,arg4f
    pslld xmm4,31
    ;xmm4[0] = [10000000000000000000000000000000]

    movss arg4f,xmm4
  
    pxor arg2f,arg4f
    pshufd arg2f,arg2f,11_10_00_01b
    mulps arg2f,arg1f

    movsd [arg2],arg2f

    leave 
    ret
%ifidn __OUTPUT_FORMAT__, win64 
    args_reset
%endif


global V2V2Dot; float DotProduct = V2V2Dot (float * Vec2_A, float * Vec2_B);
;**********************************************
;Given A and B 2D Vectors,
;it returns (A . B)
;**********************************************
V2V2Dot:
    enter 0,0
    movsd arg3f,[arg1]
    pxor arg1f,arg1f
    movsd arg2f,[arg2]
    mulps arg3f,arg2f
    movshdup arg2f,arg3f
    addss arg3f,arg2f

    movss arg1f,arg3f ;Return is stored in arg1f
    
    leave
    ret
    

global M4x4MUL ;(float * A, float *B, float * Result);
;**********************************************
;Given A and B (both 4x4 Matrices),
;4x4 Matrix Result = A * B;
;**********************************************
M4x4MUL:
    enter 0,0
        movups  arg1f, [arg1]
        movups  arg2f, [arg1+16]
        movups  arg3f, [arg1+32]
        movups  arg4f, [arg1+16+32]
    TRANS44; Matrix A (rows) in 0,1,4 and 5
	  movaps arg3f,xmm4
	  movaps arg4f,xmm5
    ;Matriz A (rows) in 0,1,2,3

        movups  xmm4, [arg2]
        movups  xmm5, [arg2+16]
        movups  xmm6, [arg2+32]
        movups  xmm7, [arg2+16+32]
    ;Matriz B (Columns) in 4,5,6,7

    M4x4MULMACRO xmm4,xmm12
    M4x4MULMACRO xmm5,xmm13
    M4x4MULMACRO xmm6,xmm14
    M4x4MULMACRO xmm7,xmm15

    movups [arg3],xmm12
    movups [arg3+16],xmm13
    movups [arg3+32],xmm14
    movups [arg3+16+32],xmm15

    leave
    ret


global NormalizeDoubleVec2; NormalizeVec2(double * A, double * Result)
;******************************************************
;Given a 2D Vector A whose elements are 64-bits long,
;this funtion returns the normalized Vector
;******************************************************
NormalizeDoubleVec2:
    enter 0,0
        movupd arg1f,[arg1]
        movapd arg2f,arg1f
        mulpd arg2f,arg1f
        movhlps arg3f,arg2f
        addsd arg2f,arg3f
        sqrtsd arg2f,arg2f
        punpcklqdq arg2f,arg2f
        divpd arg1f,arg2f
        movupd [arg2],arg1f
    leave
    ret

global NormalizeVec2; NormalizeVec2(Float * A, float * Result)
;******************************************************
;Given a 2D Vector A whose elements are 32-bits long,
;this funtion returns the normalized Vector
;******************************************************
NormalizeVec2:
    enter 0,0
        movq arg1f,[arg1]
        movq arg2f,arg1f
        mulps arg2f,arg1f
        movshdup arg3f,arg2f
        addss arg2f,arg3f
        sqrtss arg2f,arg2f
        movsldup arg2f,arg2f
        divps arg1f,arg2f
        movq [arg2],arg1f
    leave
    ret


global NormalizeVec3; NormalizeVec3(Float * A, float * Result)
;******************************************************
;Given a 3D Vector A whose elements are 32-bits long,
;this funtion returns the normalized Vector
;******************************************************
NormalizeVec3:
    enter 0,0
        movups arg1f,[arg1]
        ;arg1f [][z][y][x]
        movaps arg2f,arg1f
        ;arg2f [][z][y][x]
        mulps  arg2f,arg1f
        ;arg2f [][z*z][y*y][x*x]
        movaps arg4f,arg1f
        ;arg4f [][z][y][x]
        movhlps arg1f,arg2f
        ;arg1f [][][][z*z]
        pshufd arg3f,arg2f,1
        ;arg3f [][][][y*y]
        addss arg2f,arg3f
        addss arg2f,arg1f
        ;arg2f [][][][(x*x)+(y*y)+(z*z)]
        sqrtss arg2f,arg2f
        ;arg2f [][][][sqrt((x*x)+(y*y)+(z*z))]
        movlps [arg2],arg4f
        movss [arg2+8],arg2f
    leave
    ret

global NormalizeVec4; NormalizeVec3(Float * A, float * Result)
;******************************************************
;Given a 4D Vector A whose elements are 32-bits long,
;this funtion returns the normalized Vector
;******************************************************
NormalizeVec4:
    enter 0,0
        movups arg1f,[arg1]
        ;arg1f [w][z][y][x]
        movaps arg2f,arg1f
        ;arg2f [w][z][y][x]
        mulps  arg2f,arg1f
        ;arg2f [w*w][z*z][y*y][x*x]
        movshdup arg3f,arg2f
        addps    arg2f,arg3f
        movhlps  arg3f,arg2f
        addss    arg2f,arg3f
        sqrtss   arg2f,arg2f
        pshufd   arg2f,arg2f,0
        divps    arg1f,arg2f
        movups [arg2],arg1f
    leave
    ret

global CrossProductVec3; CrossProductVec3(float * A, float * B, float * Result)
;******************************************************
;Given A and B (both 3D Vectors)
;3D Vector Result = (A x B)
;******************************************************
CrossProductVec3:
    enter 0,0
        movups arg1f,[arg1]
        ;arg1f [?][Az][Ay][Ax]
        movups arg2f,[arg2]
        ;arg2f [?][Bz][By][Bx]
        pshufd arg3f,arg1f,11010010b
        pshufd arg4f,arg2f,11001001b
        pshufd arg1f,arg1f,11001001b
        pshufd arg2f,arg2f,11010010b
        ;arg1f [?][Bx][Bz][By]
        ;arg2f [?][By][Bx][Bz]
        ;arg3f [?][Ay][Ax][Az]
        ;arg4f [?][Bx][Bz][By]
        mulps arg1f,arg2f
        mulps arg3f,arg4f
        subps arg1f,arg3f
        ;arg1f [?][Rz][Ry][Rx]
        movhlps arg2f,arg1f
        movlps [arg3],arg1f
        movss [arg3+8],arg2f
    leave
    ret

global M4x4V4MUL;M4x4V4MUL(float * MatrixA, float *VectorB, float * Result);
;******************************************************
; Given a 4x4 Matrix MatrixA and a 4D Vector VectorB,
; 4D Vector Result = MatrixA * VectorB;
;******************************************************
M4x4V4MUL: 
    enter 0,0
        movups  arg1f, [arg1]
        movups  arg2f, [arg1+16]
        movups  arg3f, [arg1+32]
        movups  arg4f, [arg1+16+32]
    TRANS44
    movups xmm7,[arg2]
	movaps arg3f,xmm4
	movaps arg4f,xmm5
    M4x4MULMACRO xmm7,xmm15
	movups [arg3],xmm15
    leave
    ret

; 1.0f = 00111111100000000000000000000000

global M4x4V4_PseudoV3_W1_MUL
; M4x4V4_PseudoV3_W1_MUL(float * MatrixA, float *VectorB, float * Result);
;*******************************************************************************************
; Given a 4x4 Matrix MatrixA and a 3D Vector VectorB,
; 4D Vector Result = MatrixA * VectorB, BUT:
; - In order for the operation to be possible, a 4th element (1.f) is added to VectorB
;*******************************************************************************************
M4x4V4_PseudoV3_W1_MUL: 
    enter 0,0
        movups  arg1f, [arg1]
        movups  arg2f, [arg1+16]
        movups  arg3f, [arg1+32]
        movups  arg4f, [arg1+16+32]
    TRANS44
    movups xmm7,[arg2]
	movaps arg3f,xmm4
	movaps arg4f,xmm5

    pxor xmm4,xmm4
    pcmpeqd xmm4,xmm4
    ;xmm4[0] = [11111111111111111111111111111111]
    psrld xmm4,25
    ;xmm4[0] = [00000000000000000000000001111111]
    pslld xmm4,23
    ;xmm4[0] = [00111111100000000000000000000000] = 1.f
    ;xmm4 = [1.f][1.f][1.f][1.f]


    movhlps xmm5,xmm7
    ;xmm5 = [?][?][W][Z]
    movss xmm4,xmm5
    ;xmm4 = [1.f][1.f][1.f][Z]
    movlhps xmm7,xmm4
    ;xmm7 = [1.f][Z][Y][X]
    movlhps xmm5,xmm5
    ;xmm5 = [W][Z][?][?]


    M4x4MULMACRO xmm7,xmm15

    movhlps xmm5,xmm15
    ;xmm5 = [W][Z][Rw][Rz]
    pshufd xmm5,xmm5,11_10_11_00b
    ;xmm5 = [W][Z][W][Rz]
    movlhps xmm15,xmm5
    ;xmm15 = [W][Rz][Ry][Rx]

	movups [arg3],xmm15
    leave
    ret

global M4x4_PseudoM3x3_V4_PseudoV3_W1_MUL
;M4x4_PseudoM3x3_V4_PseudoV3_W1_MUL(float * MatrixA, float *VectorB, float * Resultado);
;****************************************************************************************
; Given a 4x4 Matrix MatrixA and a 3D Vector VectorB,
; 4D Vector Result = MatrixA * VectorB, BUT:
; - In order for the operation to be possible, a 4th element (1.f) is added to VectorB
; - The last row and column of the matrix are temporarily replaced both with (0,0,0,1.f)
; - This function calculates only the affine transformation.
;****************************************************************************************
M4x4_PseudoM3x3_V4_PseudoV3_W1_MUL: 
     enter 0,0
        movups  arg1f, [arg1]
        movups  arg2f, [arg1+16]
        movups  arg3f, [arg1+32]

    pxor xmm14,xmm14
    pcmpeqd xmm14,xmm14
    ;xmm14[0] = [11111111111111111111111111111111]
    psrld xmm14,25
    ;xmm14[0] = [00000000000000000000000001111111]
    pslld xmm14,23
    ;xmm14[0] = [00111111100000000000000000000000] = 1.f
    ;xmm14 = [1.f][1.f][1.f][1.f]

    pxor arg4f,arg4f
    movss arg4f,xmm14
    pshufd arg4f,arg4f, 00_10_01_11b

    TRANS44
    movups xmm7,[arg2]
	  movaps arg3f,xmm4

    pxor arg4f,arg4f
    movss arg4f,xmm14

    movhlps xmm5,xmm7
    ;xmm5 = [?][?][W][Z]
    movss xmm14,xmm5
    ;xmm14 = [1.f][1.f][1.f][Z]
    movlhps xmm7,xmm14
    ;xmm7 = [1.f][Z][Y][X]
    movlhps xmm5,xmm5
    ;xmm5 = [W][Z][?][?]


    M4x4MULMACRO xmm7,xmm15

    movhlps xmm5,xmm15
    ;xmm5 = [W][Z][Rw][Rz]
    pshufd xmm5,xmm5,11_10_11_00b
    ;xmm5 = [W][Z][W][Rz]
    movlhps xmm15,xmm5
    ;xmm15 = [W][Rz][Ry][Rx]

	  movups [arg3],xmm15
    leave
    ret

global QuaternionMUL; (float * A, float * B, float * Result);
;**********************************
; Given A and B (both Quaternions),
; Quaternion Result = A * B;
;**********************************
QuaternionMUL:
    enter 0,0
    mov arg4, SignChange32bits
    push arg4
        movss xmm5,[rsp]


        movups xmm7,[arg1]
        ;xmm7 = [Aw][Az][Ay][Ax]

        movups xmm6,[arg2]
        ;xmm7 = [Bw][Bz][By][Bx]

        pshufd xmm4,xmm6,00_01_10_11b
        ;xmm4 = [Bx][By][Bz][Bw]

        movaps arg1f,xmm4
        mulps arg1f,xmm7

        pshufd xmm4,xmm6,01_00_11_10b
        ;xmm4 = [By][Bx][Bw][Bz]

        movaps arg2f,xmm4
        mulps arg2f,xmm7

        pshufd xmm4,xmm6,10_11_00_01b
        ;xmm4 = [Bz][Bw][Bx][By]

        movaps arg3f,xmm4
        mulps arg3f,xmm7

        movaps arg4f,xmm6
        mulps arg4f,xmm7

        pshufd xmm4,xmm5,10_11_00_01b
        pxor arg1f,xmm4

        pshufd xmm4,xmm5,11_00_11_01b
        pxor arg2f,xmm4

        pshufd xmm4,xmm5,11_11_11_00b
        pxor arg3f,xmm4

        pshufd xmm4,xmm5,11_00_00_00b
        pxor arg4f,xmm4

    TRANS44

        addps arg1f,arg2f
        addps arg1f,xmm4
        addps arg1f,xmm5
        movups [arg3],arg1f
    add rsp,8

    leave
    ret

global QuaternionToMatrix4x4; QuaternionToMatrix4x4(float * Quaternion, float * Matrix)
;********************************************
;Given a Quaternion, his function generates a 4x4 Matrix.
;This algorithm is an implementation of a method by Jay Ryness (2008)
;Source: https://sourceforge.net/p/mjbworld/discussion/122133/thread/c59339da/
;********************************************
QuaternionToMatrix4x4: 
    enter 0,0
    
    mov dword[arg2],SignChange32bits
                movss xmm7,[arg2]
                ;xmm7 [0][0][0][0x800000]

    sub rsp,16
                pshufd arg1f,xmm7,00_11_11_11b
                movups arg4f,[arg1]
                movups arg2f,arg4f
                ;xmmm3 [w][z][y][x]
                movaps arg3f,xmm7
                ;xmm7 [0][0][0][0x800000]
                pxor arg4f,arg1f
                pshufd xmm7,xmm7,0h
                movups [rsp],arg4f
                ;xmm7 [0x800000][0x800000][0x800000][0x800000]
                pshufd arg3f,arg3f,11110000b
                ;arg3f [0][0][0x800000][0x800000]
    mov arg3,rsp
                movups arg4f,arg2f
                ;xmmm3 [w][z][y][x]
        fld dword[arg3]
    add arg3,4
        fld dword[arg3]
                pxor xmm7,arg4f
                ;xmmm7 [-w][-z][-y][-x]
    add arg3,4
        fld dword[arg3]
    add arg3,4
        fld dword[arg3]
        ;Stack: ST0= -w; ST1=z; ST2=y; ST3=x
        
	;arg4f= [w][z][y][x]
        ;xmm7= [-w][-z][-y][-x]
                movups [arg2+16+16+16],xmm7
                pshufd arg2f,arg4f,01111110b
                ;arg2f [Y][W][W][Z]
                movaps arg1f,arg2f
                ;arg1f [Y][W][W][Z]
                punpcklqdq arg2f,xmm7
                ;arg2f [-Y][-X][W][Z]
                movups [arg2+16],arg2f

                pshufd xmm6,xmm7,00100100b
                ;xmm6 [-X][-Z][-Y][-X]
                ;arg1f [Y][W][W][Z]
                punpckhdq arg1f,xmm6
                ;arg1f [-X][Y][-Z][W]
                movaps xmm5,arg1f
                ;xmm5 [-X][Y][-Z][W]
                movups [arg2],arg1f

                ;xmm5 [-X][Y][-Z][W]
                pshufd xmm5,xmm5,01_00_11_10b
                ;xmm5 [-Z][W][-X][Y]
                pxor arg3f,xmm5
                ;xmm5 [-Z][W][X][-Y]
                movups [arg2+16+16],arg3f

        fchs
        fstp dword [arg2+16+16+16+12]
        fstp dword [arg2+16+16+16+12-16]
        fstp dword [arg2+16+16+16+12-16-16]
        fstp dword [arg2+16+16+16+12-16-16-16]

    TRANS44
    
    mov arg1,arg2
    xor arg3,arg3
    mov arg4,arg2
    Mmullabelqua:
        MULVEC4VEC4 arg1,arg4,arg3
        add arg3,16
        cmp arg3,64
        jne Mmullabelqua

    add rsp,16
    leave
    ret

global AxisAngleToQuaternion; AxisAngleToQuaternion(float * Axis, float Degree, float * Quaternion)
;*********************************************************************
;Given a 3D Vector describing an Axis and a angle given in degrees,
;This function calculates the respective quaternion.
;*********************************************************************
%ifidn __OUTPUT_FORMAT__, win64 
    %define arg1f XMM1
    %define arg2 R8
%endif
AxisAngleToQuaternion:
    enter 0,0
	sub rsp,8
	movss [rsp],arg1f
        push fc_180fdivPI
        fld dword [rsp+8]
        fld dword [rsp]
        fdivp
        fld1
    add rsp,16
        fld1

        faddp
        fdivp
        fld st0
            movups arg1f,[arg1]
        fcos
        fxch

        fsin
        fstp dword[arg2]
            movss XMM3,[arg2]
            pshufd XMM3,XMM3,0h
            mulps arg1f,XMM3
            movups [arg2],arg1f
        fstp dword[arg2+4+4+4]
    leave
    ret
%ifidn __OUTPUT_FORMAT__, win64 
    args_reset
%endif

global PerspectiveProjectionMatrix4x4
;(float *matrix, float fovyInDegrees, float aspectRatio,float znear, float zfar);
;*********************************************************************
;It's an implementation of gluPerspective
;*********************************************************************
PerspectiveProjectionMatrix4x4: 

enter 0,0

%ifidn __OUTPUT_FORMAT__, win64 
    %define arg1f XMM1
    %define arg2f XMM2
    %define arg3f XMM3
    %define arg4f XMM4

    ;zfar is in the stack, so it must be move to XMM4

    movups xmm4,arg5
%endif

    mov rax, fc_360f
	    pxor xmm12,xmm12
	    movaps xmm11,arg4f
push rax
	    mov rax, fc_m_1f
	    pxor xmm10,xmm10
	fldpi
sub rsp,8
	fstp dword[rsp]
	    movss xmm12,arg3f
	    pxor  xmm9,xmm9
	    movss xmm13,[rsp]
	    addss xmm12,xmm12
	    subss xmm11,arg3f
	    movss xmm14,[rsp+8]
	    mulss arg1f,xmm13
	    subss xmm10,xmm12
	    divss arg1f,xmm14
	    movss [rsp],arg1f
	    mulss xmm10,arg4f
	    movss xmm9,xmm11
	fld dword [rsp]
	    divss xmm10,xmm11
	    subss xmm9,arg4f
	fptan
	fstp st0
	fstp dword [rsp]
	    subss xmm9,arg4f
	    divss xmm9,xmm11
	    pxor xmm5,xmm5
;XMM11 = temp4 = ZFAR - ZNEAR
;XMM9  = (-ZFAR - ZNEAR)/temp4
;XMM10 = (-temp * ZFAR) / temp4
;XMM12 = temp  =2.0 * ZNEAR

	    pshufd xmm7,xmm10,11_00_11_11b
	    movss arg1f, [rsp]
	    pshufd xmm6,xmm9, 11_00_11_11b
	    mulss arg1f, arg3f
	    mulss arg2f, arg1f
	    addss arg1f,arg1f
;arg1f = temp3
	    movss xmm5,xmm12
	    divss xmm5,arg1f
	    pshufd xmm5,xmm5,11_11_00_11b
	    addss arg2f,arg2f
;arg2f = temp2
	    divss xmm12,arg2f

;Resulting matrix in XMM12,XMM5,XMM6,XMM7

	    movups [arg1],XMM12
	    movups [arg1+16],XMM5
	    movups [arg1+16+16],XMM6
	    movups [arg1+16+16+16],XMM7
        mov    [arg1+16+16+12],rax

add rsp,8
    leave

%ifidn __OUTPUT_FORMAT__, win64 
args_reset
%endif

    ret


global OrthogonalProjectionMatrix4x4;
;    void OrthogonalProjectionMatrix4x4
;    (float *matrix, float Width, float Height, float znear, float zfar);
;*********************************************************

OrthogonalProjectionMatrix4x4: 

%ifidn __OUTPUT_FORMAT__, win64 
    %define arg1f XMM1
    %define arg2f XMM2
    %define arg3f XMM3
    %define arg4f XMM4

    ;zfar is in the stack, so it must be move to XMM4

    movups xmm4,arg5
%endif

    enter 0,0

    mov arg2,fc_2f
    movss xmm4,arg4f
    subss xmm4,arg3f
    push arg2
    addss arg4f,arg3f
    divss arg4f,xmm4
    movss arg3f,[rsp]
    pxor xmm5,xmm5
    pxor xmm6,xmm6
    movss xmm7,arg3f
    mov arg2,fc_1f
    divss arg3f,arg1f
    movss arg1f,xmm7
    divss xmm7,arg2f
    subss xmm6,arg1f
    subss xmm5,arg4f
    divss xmm6,xmm4

;arg3f = 2/Width
;arg4f = (zfar+znear)/(zfar-znear)
;xmm4 = zfar-znear
;xmm5 = -((zfar+znear)/(zfar-znear))
;xmm6 = -2
;xmm7 = 2/Height

    movups [arg1],arg3f
    pxor arg1f,arg1f
    pshufd xmm7,xmm7,11_11_00_11b
    divss xmm6,xmm4
    movups [arg1+16],xmm7
    pshufd xmm5,xmm5,11_00_11_11b
    pshufd xmm6,xmm6,11_00_11_11b
    movups [arg1+16+16],xmm6

    add rsp,8

    movups [arg1+16+16+16],xmm5
    mov [arg1+16+16+16+12],arg2

    leave

%ifidn __OUTPUT_FORMAT__, win64 
    args_reset
%endif

    ret

global ViewLookAt
; ViewLookAt(float * matrix, float * Vec3From_EYE, float * Vec3To_CENTER, float * Vec3Up);
;*********************************************************
;It's an implementation of glm::LookAt
;*********************************************************
ViewLookAt: 
enter 0,0
    push rax
    xor eax,eax
    mov eax,fc_m_1f
    push rax 

    pxor arg4f,arg4f

    movups xmm9, [arg2] ;EYE
    movups xmm15, [arg3] ;CENTER
    subps xmm15,xmm9 ;xmm15 = f = CENTER - EYE

    movups xmm14, [arg4]
    ;---Normalize f----;
    NORMALIZEVEC3MACRO xmm15,arg1f,arg2f,arg3f
    ;-------------------;
    ;---Normalize up----;
    NORMALIZEVEC3MACRO xmm14,arg1f,arg2f,arg3f
    ;-------------------;

    ;Resumen:
    ;xmm15 = f
    ;xmm14 = up

    movss xmm8, [rsp]

    ;Cross Product s = f x up;
    CROSSPRODUCTMACRO xmm15,xmm14,xmm13,arg1f,arg2f,arg3f
    ;--------------------------;
    ;Normalize s-----;
    NORMALIZEVEC3MACRO xmm13,arg1f,arg2f,arg3f
    ;-----------------;

    ;Resume:
    ;xmm9 = eye
    ;xmm15 = f
    ;xmm14 = up
    ;xmm13 = s

    pshufd xmm8,xmm8,0
    ;xmm8 [-1.f][-1.f][-1.f][-1.f]

    add rsp,8

    ;Cross Product u = s x f;
    CROSSPRODUCTMACRO xmm13,xmm15,xmm14,arg1f,arg2f,arg3f
    ;-------------------------;

    ;Resume:
    ;xmm9 = eye
    ;xmm15 = f
    ;xmm14 = u
    ;xmm13 = s 

    ;calculate -( s . eye )
    DotProductXMMV3 xmm13,xmm9,xmm12,arg1f
    mulss xmm12,xmm8
    ;------------------------------;

    pop rax

    ;calculate -( u . eye )
    DotProductXMMV3 xmm14,xmm9,xmm11,arg1f
    mulss xmm11,xmm8
    ;------------------------------;

    ;calculate ( f . eye )
    DotProductXMMV3 xmm15,xmm9,xmm10,arg1f
    ;------------------------------;  

    ;do f=-f;
    mulps xmm15,xmm8
    ;----------;

    ;Resume:
    ;xmm8 = [-1][-1][-1][-1]
    ;xmm9 = eye
    ;xmm15 = -f
    ;xmm14 = u
    ;xmm13 = s 
    ;xmm12 = -dot (s,eye)
    ;xmm11 = -dot (u,eye)
    ;xmm10 = +dot (f,eye)

    mulps xmm8,xmm8
    ;xmm8 = [1.f][1.f][1.f][1.f]
    movss xmm8,xmm10
    ;xmm8 = [1.f][1.f][1.f][+dot(f,eye)]
    movlhps xmm8,xmm8
    ;xmm8 = [1.f][+dot(f,eye)][1.f][+dot(f,eye)]
    unpcklps xmm12,xmm11
    ;xmm12 = [-dot (u,eye)][-dot (s,eye)][-dot (u,eye)][-dot (s,eye)]
    movsd xmm8,xmm12
    ;xmm8 [1.f][+dot(f,eye)][-dot (u,eye)][-dot (s,eye)]

    movaps arg1f,xmm13
    movaps arg2f,xmm14
    movaps arg3f,xmm15

    TRANS44

    movaps [arg1],arg1f
    add arg1,16
    movaps [arg1],arg2f
    add arg1,16
    movaps [arg1],xmm4
    add arg1,16
    movaps [arg1],xmm8

leave
ret
