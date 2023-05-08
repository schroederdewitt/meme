#include <torch/extension.h>
#include <stdlib.h>
#include <unordered_set>
#include <mpfr.h>
#include <gmpxx.h>
#include <gmp.h>
#include <functional>
#include <queue>


torch::Tensor greatest_lower_bound(torch::Tensor p, torch::Tensor q){ //# compute p hat q
    //# https://arxiv.org/pdf/1901.07530.pdf (fact 1)
    //# input: p, q of dims batch x dim
    //assert p.shape == q.shape, "p, q need to have same dimensions!"
    auto dim = p.size(1);
    auto z = p.clone().zero_();

    std::vector<torch::Tensor> tmp;
    for (int b=0;b<q.size(0);b++){
        tmp.push_back(torch::minimum(q[b][0], p[b][0]));
    }
    z.slice(1, 0, 1).copy_(torch::stack(tmp, -1));
    auto cumsum_p = torch::cumsum(p, -1);
    auto cumsum_q = torch::cumsum(q, -1);

    torch::Tensor sum_z;
    for(int i=1; i<dim;i++){
        sum_z = torch::sum(z.slice(1, 0, i), -1);
        std::vector<torch::Tensor> tmp;
        for (int b=0;b<q.size(0);b++){
            tmp.push_back(torch::minimum(cumsum_p[b][i], cumsum_q[b][i]));
        }
        z.slice(1, i, i+1).copy_(torch::stack(tmp, -1) - sum_z);
    }
    return z;
}

torch::Tensor greatest_lower_bound_fast(torch::Tensor p, torch::Tensor q){ //# compute p hat q
    //# https://arxiv.org/pdf/1901.07530.pdf (fact 1)
    //# input: p, q of dims batch x dim
    //assert p.shape == q.shape, "p, q need to have same dimensions!"
    auto dim = p.size(1);
    auto z = p.clone().zero_();

    std::vector<torch::Tensor> tmp;
    for (int b=0;b<q.size(0);b++){
        tmp.push_back(torch::minimum(q[b][0], p[b][0]));
    }
    z.slice(1, 0, 1).copy_(torch::stack(tmp, -1));
    auto cumsum_p = torch::cumsum(p, -1);
    auto cumsum_q = torch::cumsum(q, -1);
    auto minsum = cumsum_p.clone();
    auto mask = cumsum_q < cumsum_p;
    minsum.masked_scatter_(mask, torch::masked_select(cumsum_q, mask));
    torch::Tensor sum_z;
    for(int i=1; i<dim;i++){
        sum_z = torch::sum(z.slice(1, 0, i), -1);
        z.slice(1, i, i+1).copy_(minsum.slice(1,i,i+1) - sum_z);
    }
    return z;
}

std::tuple<long double, long double, std::unordered_set<unsigned int>> lltm_lemma3(long double z, long double x, torch::Tensor A, unsigned int i, unsigned int j);

torch::Tensor lltm_mec(long n, torch::Tensor M, torch::Tensor q, torch::Tensor p, torch::Tensor z, bool verbose=false, double atol=10e-8){
    double z_i_d, z_i_r, sum_m_ki, sum_m_ik;
    std::unordered_set<unsigned int> I;
    long i = n-1;
    while (i >= 0){
        sum_m_ki = M.slice(0, i, n).slice(1, i, i+1).sum().item().to<double>();
        if (verbose) std::cout<<"New i: "<<i<<std::endl;
        if (verbose) std::cout<<"Investigating branch 1 with sum "<<(sum_m_ki)<<" and q: "<< q[i].item().to<double>()<<std::endl;
        if (sum_m_ki > q[i].item().to<double>() && !torch::isclose(sum_m_ki-q[i], torch::tensor({0.0}, torch::kFloat64), atol=atol).item().to<bool>()){
            std::tie(z_i_d, z_i_r, I) = lltm_lemma3(z[i].item().to<double>(), q[i].item().to<double>(), M.slice(1, i, i+1).view(-1), 0, n-1);
            if (verbose) std::cout<<"z_i_d: "<<z_i_d<<" z_i_r: "<<z_i_r<<" i:"<<i<<std::endl;
            if (verbose) std::cout<<"Entered branch 1 with sum "<<sum_m_ki<<" and q: "<< q[i].item().to<double>()<<std::endl;
            if (verbose){
                std::cout<<"Received I: ";
                for(auto ptr0=I.begin();ptr0!=I.end();ptr0++){
                    std::cout<<*ptr0<<",";
                }
                std::cout<<std::endl;
            }
            M[i][i] = z_i_d;
            M[i][i - 1] = z_i_r;
            for (long k=0; k<n; k++){
                if (I.find(k) == I.end() && k!=i){
                    M[k][i-1] = M[k][i];
                    M[k][ i] = 0;
                }
            }
        }

        sum_m_ik = M.slice(1, i, n).slice(0, i, i+1).sum().item().to<double>();
        if (verbose) std::cout<<"Investigate branch 2 with sum "<<sum_m_ik<<" and p: "<< p[i].item().to<double>()<<std::endl;
        if(sum_m_ik > p[i].item().to<double>() && !torch::isclose(sum_m_ik - p[i], torch::tensor({0.0}, torch::kFloat64), atol=atol).item().to<bool>()){
            std::tie(z_i_d, z_i_r, I) = lltm_lemma3(z[i].item().to<double>(), p[i].item().to<double>(), M.slice(0, i, i+1).view(-1), 0, n-1);
            if (verbose) std::cout<<"z_i_d: "<<z_i_d<<" z_i_r: "<<z_i_r<<" i:"<<i<<std::endl;
            if (verbose) std::cout<<"Entered branch 2 with sum "<<sum_m_ik<<" and p: "<< p[i].item().to<double>()<<std::endl;
            if (verbose){
                std::cout<<"Received I: ";
                for(auto ptr0=I.begin();ptr0!=I.end();ptr0++){
                    std::cout<<*ptr0<<",";
                }
                std::cout<<std::endl;
            }
            M[i][i] = z_i_d;
            M[i - 1][i] = z_i_r;

            for (long k=0; k<n; k++){
                if (I.find(k) == I.end() && k!=i){
                    M[i-1][k] = M[i][k];
                    M[i][k] = 0;
                }
            }
            //std::cout<<M;
        }
        i = i - 1;
    }
    if(verbose) std::cout<<"Final round......"<<std::endl;
    return M;
}

std::tuple<long double, long double, std::unordered_set<unsigned int>> lltm_lemma3(long double z, long double x, torch::Tensor A, unsigned int i, unsigned int j){
    unsigned int k = i ;
    std::unordered_set<unsigned int> I;
    long double sum = 0.0;

    while(true){
        if(A.slice(0, 0, k).sum().item().to<double>() > x) break;
        I.insert(k);
        k = k + 1;
        if(k > j) break;
    }
    long double z_r = A.slice(0, 0, k).sum().item().to<double>() - x; // - x;
    long double z_d = z - z_r;
    return std::make_tuple(z_d, z_r, I);
}


struct nested_pair_hash
{
    template <class T1, class T2, class T3>
    std::size_t operator () (std::pair<T1, std::pair<T2, T3>> const &pair) const
    {
        std::size_t h1 = std::hash<T1>()(pair.first);
        std::size_t h2 = std::hash<T2>()(pair.second.first);
        std::size_t h3 = std::hash<T3>()(pair.second.second);
        return h1 ^ h2 ^ h3;
    }
};


struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator () (std::pair<T1, T2> const &pair) const
    {
        std::size_t h1 = std::hash<T1>()(pair.first);
        std::size_t h2 = std::hash<T2>()(pair.second);

        return h1 ^ h2;
    }
};

auto cmper = [](std::pair<long double, long> pair1, std::pair<long double, long> pair2) { return pair1.first > pair2.first;};
typedef std::priority_queue<std::pair<long double, long>, std::vector<std::pair<long double, long>>, decltype(cmper)> queuetype;
std::tuple<long double, long double, std::vector<std::pair<long double, long>>, long double> lltm_lemma3_sparse(long double z,
                                              long double x,
                                              std::priority_queue<std::pair<long double, long>> &Q,
                                              long double qsum);



torch::Tensor lltm_mec_sparse(torch::Tensor pr, torch::Tensor qr, torch::Tensor zr, bool verbose=false, double atol=1e-6){
    // Create priority queues

    //TODO: VERIFY WE HAVE A MIN QUEUE!

    std::vector<long double> p;
    std::vector<long double> q;
    std::vector<long double> z;
    for(long i=0;i<pr.numel();i++){
        p.push_back(static_cast<long double>(pr[i].item().to<double>()));
        q.push_back(static_cast<long double>(qr[i].item().to<double>()));
        z.push_back(static_cast<long double>(zr[i].item().to<double>()));
    }

    std::priority_queue<std::pair<long double, long>> Q_row;
    std::priority_queue<std::pair<long double, long>> Q_col;

    long double qcolsum = 0.0;
    long double qrowsum = 0.0;
    long double z_i_d, z_i_r;
    std::vector<std::pair<long double, long>> I;
    std::vector<std::pair<long double, std::pair<long, long>>> L;



    long n = z.size();
    for(long i=n-1; i>=0; i--){
        if(verbose) std::cout<<"New i: "<<i<<std::endl;
        z_i_d = z[i];
        z_i_r = 0.0;

        if (verbose) std::cout<<"Investigating branch 1 with sum "<<(qcolsum + z[i])<<" and q: "<< q[i]<<std::endl;
        if(qcolsum + z[i] >= q[i] + atol){
            std::tie(z_i_d, z_i_r, I, qcolsum) = lltm_lemma3_sparse(z[i],
                                                                    q[i],
                                                                    Q_col,
                                                                    qcolsum);
            if (verbose) std::cout<<"z_i_d: "<<z_i_d<<" z_i_r: "<<z_i_r<<" i:"<<i<<std::endl;
            if (verbose) std::cout<<"Entered branch 1 with sum "<<(qcolsum + z[i])<<" and q: "<< q[i]<<std::endl;
            double m;
            long l;
            if (verbose){
                std::cout<<"Received I: ";
                for(auto ptr0=I.begin();ptr0!=I.end();ptr0++){
                    std::cout<<*ptr0<<",";
                }
                std::cout<<std::endl;
            }
            for(auto itr=I.begin();itr!=I.end();itr++){
                std::tie(m, l) = *itr;
                L.push_back(std::make_pair(m, std::make_pair(l, i)));
                if (verbose) std::cout<<"Pushing back... "<<std::make_pair(m, std::make_pair(l, i))<<std::endl;
            }
            if(z_i_r > -atol) {
                Q_col.push(std::make_pair(z_i_r, i));
                qcolsum = qcolsum + z_i_r;
                if (verbose) std::cout<<"Adding z_i_r..."<<std::endl;
            }
        } 
        else if(qcolsum + z[i] >= q[i] - atol) {
            long double m; long l;
            if(verbose) std::cout<<"DRAW EVEN"<<std::endl;
            while(Q_col.size()){
                std::tie(m,l) = Q_col.top();
                Q_col.pop();
                qcolsum = qcolsum - m;
                L.push_back(std::make_pair(m, std::make_pair(l, i)));
                if (verbose) std::cout<<"Pushing back... "<<std::make_pair(m, std::make_pair(l, i))<<std::endl;
            }
        }

        if (verbose) std::cout<<"Investigate branch 2 with sum "<<(qrowsum + z[i])<<" and p: "<< p[i]<<std::endl;
        if(qrowsum + z[i] > p[i] + atol){
            std::tie(z_i_d, z_i_r, I, qrowsum) = lltm_lemma3_sparse(z[i],
                                                                    p[i],
                                                                    Q_row,
                                                                    qrowsum);
            if (verbose) std::cout<<"Entered branch 2 with sum "<<(qrowsum + z[i])<<" and p: "<< p[i]<<std::endl;
            double m; long l;
            if (verbose){
                std::cout<<"Received I: ";
                for(auto ptr0=I.begin();ptr0!=I.end();ptr0++){
                    std::cout<<*ptr0<<",";
                }
                std::cout<<std::endl;
            }
            for(auto itr=I.begin();itr!=I.end();itr++){
                std::tie(m, l) = *itr;
                L.push_back(std::make_pair(m, std::make_pair(i, l)));
                if (verbose) std::cout<<"Pushing back... "<<std::make_pair(m, std::make_pair(i, l))<<std::endl;
            }

            if(z_i_r > -atol) {
                Q_row.push(std::make_pair(z_i_r, i));
                qrowsum = qrowsum + z_i_r;
                if (verbose) std::cout<<"Adding z_i_r..."<<std::endl;
            }
        } 
        else if (qrowsum + z[i] >= p[i] - atol){
                double m; long l;
                if(verbose) std::cout<<"DRAW EVEN"<<std::endl;
                while(Q_row.size()){
                    std::tie(m,l) = Q_row.top();
                    Q_row.pop();
                    qrowsum = qrowsum - m;
                    L.push_back(std::make_pair(m, std::make_pair(i, l)));
                    if (verbose) std::cout<<"Pushing back... "<<std::make_pair(m, std::make_pair(i, l))<<std::endl;
                }
            }
        if (verbose) std::cout<<"Pushing back at loop end... "<<std::make_pair(z_i_d, std::make_pair(i, i))<<std::endl;
        L.push_back(std::make_pair(z_i_d, std::make_pair(i, i)));
    }

    auto dense_ret = torch::zeros({p.size(), q.size()}, torch::kFloat64);
    for (auto h=L.begin();h!=L.end();h++){
        if(verbose) std::cout<<"v: "<<h->first<<" x:"<<h->second.first<<" y:"<<h->second.second<<std::endl;
        dense_ret[h->second.first][h->second.second] = (double) h->first;
    }

}

std::tuple<long double, long double, std::vector<std::pair<long double, long>>, long double> lltm_lemma3_sparse(long double z,
                                              long double x,
                                              std::priority_queue<std::pair<long double, long>> &Q,
                                              long double qsum){
    std::vector<std::pair<long double, long>> I;
    long double sum = 0.0;
    long double m; long l;
    while(Q.size() && sum + Q.top().first < x){
        std::tie(m, l) = Q.top();
        Q.pop();
        qsum = qsum - m;
        I.push_back(std::make_pair(m, l));
        sum = sum + m;
    }
    long double z_d = x - sum;
    long double z_r = z - z_d;
    return std::make_tuple(z_d, z_r, I, qsum);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("glb", &greatest_lower_bound, "LLTM glb");
  m.def("glbfast", &greatest_lower_bound_fast, "LLTM glb fast");
  m.def("mec", &lltm_mec, "LLTM mec");
  m.def("lemma3", &lltm_lemma3, "LLTM Lemma 3");
  m.def("mec_sparse", &lltm_mec_sparse, "LLTM mec sparse");
  m.def("lemma3_sparse", &lltm_lemma3_sparse, "LLTM Lemma 3 sparse");
}
