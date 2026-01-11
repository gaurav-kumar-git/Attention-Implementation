#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>

// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int x,   int y, int z, int &b, const int &sizeX, const int &sizeY, const int &sizeZ) {
            
    return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int x, int y, int z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
        tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * (sizeZ) + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {

            // Compute Q * K^T
            // Q is (N, d), K is (N, d). Result QK_t is (N, N)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < d; k++) {
                        // Dot product of Row i of Q and Row j of K
                        sum += fourDimRead(Q, b, h, i, k, H, N, d) * 
                               fourDimRead(K, b, h, j, k, H, N, d);
                    }
                    twoDimWrite(QK_t, i, j, N, sum);
                }
            }
            
            // Row-wise Softmax on QK_t
            for (int i = 0; i < N; i++) {
                float row_sum = 0;
                // Calculate exponentials and sum them
                for (int j = 0; j < N; j++) {
                    float val = exp(twoDimRead(QK_t, i, j, N));
                    twoDimWrite(QK_t, i, j, N, val);
                    row_sum += val;
                }
                // Divide by sum to normalize
                for (int j = 0; j < N; j++) {
                    float val = twoDimRead(QK_t, i, j, N) / row_sum;
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }

            // Compute (Softmax Result) * V
            // QK_t is (N, N), V is (N, d). Result O is (N, d)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < d; j++) {
                    float sum = 0;
                    for (int k = 0; k < N; k++) {
                        sum += twoDimRead(QK_t, i, k, N) * 
                               fourDimRead(V, b, h, k, j, H, N, d);
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, sum);
                }
            }
        }
    }
    
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    // -------- YOUR CODE HERE  -------- //
    const int BLOCK = 32;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {

            // Blocked MatMul: Q * K^T
            // Result QK_t is (N, N). Q is (N, d), K is (N, d)
            // Initialize QK_t to 0 for this head
            for (int i = 0; i < N * N; i++) QK_t[i] = 0;

            for (int i_out = 0; i_out < N; i_out += BLOCK) {
                for (int j_out = 0; j_out < N; j_out += BLOCK) {
                    for (int k_out = 0; k_out < d; k_out += BLOCK) {
                        
                        // Internal loops over the tile
                        for (int i = i_out; i < std::min(i_out + BLOCK, N); i++) {
                            for (int j = j_out; j < std::min(j_out + BLOCK, N); j++) {
                                float sum = 0;
                                for (int k = k_out; k < std::min(k_out + BLOCK, d); k++) {
                                    sum += fourDimRead(Q, b, h, i, k, H, N, d) * 
                                           fourDimRead(K, b, h, j, k, H, N, d);
                                }
                                // Accumulate into QK_t
                                float current = twoDimRead(QK_t, i, j, N);
                                current += sum;
                                twoDimWrite(QK_t, i, j, N, current);
                            }
                        }
                    }
                }
            }

            // Unfused Softmax
            for (int i = 0; i < N; i++) {
                float row_sum = 0;
                for (int j = 0; j < N; j++) {
                    float val = exp(twoDimRead(QK_t, i, j, N));
                    twoDimWrite(QK_t, i, j, N, val);
                    row_sum += val;
                }
                for (int j = 0; j < N; j++) {
                    float val = twoDimRead(QK_t, i, j, N) / row_sum;
                    twoDimWrite(QK_t, i, j, N, val);
                }
            }

            // QK_t is (N, N), V is (N, d). Result O is (N, d)
            for (int i_out = 0; i_out < N; i_out += BLOCK) {
                for (int j_out = 0; j_out < d; j_out += BLOCK) {
                    for (int k_out = 0; k_out < N; k_out += BLOCK) {
                        
                        for (int i = i_out; i < std::min(i_out + BLOCK, N); i++) {
                            for (int j = j_out; j < std::min(j_out + BLOCK, d); j++) {
                                float sum = 0;
                                for (int k = k_out; k < std::min(k_out + BLOCK, N); k++) {
                                    sum += twoDimRead(QK_t, i, k, N) * 
                                           fourDimRead(V, b, h, k, j, H, N, d);
                                }
                                float current = fourDimRead(O, b, h, i, j, H, N, d);
                                current += sum;
                                fourDimWrite(O, b, h, i, j, H, N, d, current);
                            }
                        }
                    }
                }
            }
        }
    }


    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //

    #pragma omp parallel for collapse(2) // <- Enable OpenMP Parallelization(this is needed for paralelism and is instructed to do so here)

    // We give you a template of the first three loops for your convenience
    //loop over batch
    for (int b = 0; b < B; b++){

        //loop over heads
        for (int h = 0; h < H; h++){
            for (int i = 0; i < N ; i++){

		// YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);
		//YOUR CODE HERE
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < d; k++) {
                        sum += fourDimRead(Q, b, h, i, k, H, N, d) * 
                               fourDimRead(K, b, h, j, k, H, N, d);
                    }
                    ORow[j] = sum;
                }

                // 2. Apply Softmax to the row buffer
                float row_sum = 0.0f;
                for (int j = 0; j < N; j++) {
                    float exp_val = exp(ORow[j]);
                    ORow[j] = exp_val;
                    row_sum += exp_val;
                }
                for (int j = 0; j < N; j++) {
                    ORow[j] /= row_sum;
                }

                // 3. Multiply Softmax row by V and write to Output O
                for (int j = 0; j < d; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < N; k++) {
                        sum += ORow[k] * 
                               fourDimRead(V, b, h, k, j, H, N, d);
                    }
                    fourDimWrite(O, b, h, i, j, H, N, d, sum);
                }

            }
	}
    }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            
            // THREAD-LOCAL denominator: one value per token row
            std::vector<float> local_l(N, 0.0f); 
            // THREAD-LOCAL block buffer for exponentials
            std::vector<float> local_Pij(Br * Bc, 0.0f);

            // Step 1: Outer loop over blocks of K, V (Columns)
            for (int j_block = 0; j_block < N; j_block += Bc) {
                int cur_Bc = std::min(Bc, N - j_block);

                // Step 2: Inner loop over blocks of Q, O (Rows)
                for (int i_block = 0; i_block < N; i_block += Br) {
                    int cur_Br = std::min(Br, N - i_block);

                    // A. Compute Sij = Qi * Kj^T and store exp(Sij) in local_Pij
                    for (int i = 0; i < cur_Br; i++) {
                        for (int j = 0; j < cur_Bc; j++) {
                            float dot = 0.0f;
                            for (int k = 0; k < d; k++) {
                                dot += fourDimRead(Q, b, h, i_block + i, k, H, N, d) * 
                                       fourDimRead(K, b, h, j_block + j, k, H, N, d);
                            }
                            local_Pij[i * Bc + j] = exp(dot);
                        }
                    }

                    // B. Update the Output and Denominator for this block
                    for (int i = 0; i < cur_Br; i++) {
                        float row_sum_Pij = 0.0f;
                        for (int j = 0; j < cur_Bc; j++) {
                            row_sum_Pij += local_Pij[i * Bc + j];
                        }
                        
                        float l_old = local_l[i_block + i];
                        float l_new = l_old + row_sum_Pij;

                        // C. Update Output row using the incremental softmax rule
                        for (int dim = 0; dim < d; dim++) {
                            float p_v_product = 0.0f;
                            for (int j = 0; j < cur_Bc; j++) {
                                p_v_product += local_Pij[i * Bc + j] * 
                                               fourDimRead(V, b, h, j_block + j, dim, H, N, d);
                            }

                            float o_old = fourDimRead(O, b, h, i_block + i, dim, H, N, d);
                            // Correct Flash Attention update math:
                            float o_new = (l_old * o_old + p_v_product) / l_new;
                            
                            fourDimWrite(O, b, h, i_block + i, dim, H, N, d, o_new);
                        }

                        // Save the new denominator for the next j_block
                        local_l[i_block + i] = l_new;
                    }
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
