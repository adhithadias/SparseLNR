#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/type.h"
#include "taco/index_notation/kernel.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/transformations.h"

using namespace taco;
const IndexVar i("i"), j("j"), k("k"), l("l"), m("m");

TEST(indexstmt, assignment) {
  Type t(type<double>(), {3});
  TensorVar a("a", t), b("b", t), c("c", t);

  IndexStmt stmt = a(i) = b(i) + c(i);
  ASSERT_TRUE(isa<Assignment>(stmt));
  Assignment assignment = to<Assignment>(stmt);
  ASSERT_TRUE(equals(a(i), assignment.getLhs()));
  ASSERT_TRUE(equals(b(i) + c(i), assignment.getRhs()));
  ASSERT_EQ(IndexExpr(), assignment.getOperator());
}

TEST(indexstmt, forall) {
  Type t(type<double>(), {3});
  TensorVar a("a", t), b("b", t), c("c", t);

  IndexStmt stmt = forall(i, a(i) = b(i) + c(i));
  ASSERT_TRUE(isa<Forall>(stmt));
  Forall forallstmt = to<Forall>(stmt);
  ASSERT_EQ(i, forallstmt.getIndexVar());
  ASSERT_TRUE(equals(a(i) = b(i) + c(i), forallstmt.getStmt()));
  ASSERT_TRUE(equals(forall(i, a(i) = b(i) + c(i)), forallstmt));
}

TEST(indexstmt, where) {
  Type t(type<double>(), {3});
  TensorVar a("a", t, Sparse), b("b", t, Sparse), c("c", t, Sparse);
  TensorVar w("w", t, Dense);

  IndexStmt stmt = where(forall(i, a(i)=w(i)*c(i)), forall(i, w(i)=b(i)));
  ASSERT_TRUE(isa<Where>(stmt));
  Where wherestmt = to<Where>(stmt);
  ASSERT_TRUE(equals(forall(i, a(i)=w(i)*c(i)), wherestmt.getConsumer()));
  ASSERT_TRUE(equals(forall(i, w(i)=b(i)), wherestmt.getProducer()));
  ASSERT_TRUE(equals(where(forall(i, a(i)=w(i)*c(i)), forall(i, w(i)=b(i))),
                     wherestmt));
}

TEST(indexstmt, multi) {
  Type t(type<double>(), {3});
  TensorVar a("a", t, Sparse), b("b", t, Sparse), c("c", t, Sparse);

  IndexStmt stmt = multi(a(i)=c(i), b(i)=c(i));
  ASSERT_TRUE(isa<Multi>(stmt));
  Multi multistmt = to<Multi>(stmt);
  ASSERT_TRUE(equals(multistmt.getStmt1(), a(i) = c(i)));
  ASSERT_TRUE(equals(multistmt.getStmt2(), b(i) = c(i)));
  ASSERT_TRUE(equals(multistmt, multi(a(i)=c(i), b(i)=c(i))));
}

TEST(indexstmt, sequence) {
  Type t(type<double>(), {3});
  TensorVar a("a", t, Sparse), b("b", t, Sparse), c("c", t, Sparse);

  IndexStmt stmt = sequence(a(i) = b(i), a(i) += c(i));
  ASSERT_TRUE(isa<Sequence>(stmt));
  Sequence sequencestmt = to<Sequence>(stmt);
  ASSERT_TRUE(equals(a(i) = b(i), sequencestmt.getDefinition()));
  ASSERT_TRUE(equals(a(i) += c(i), sequencestmt.getMutation()));
  ASSERT_TRUE(equals(sequence(a(i) = b(i), a(i) += c(i)),
                     sequencestmt));
}

TEST(indexstmt, spmm) {
  Type t(type<double>(), {3,3});
  TensorVar A("A", t, Sparse), B("B", t, Sparse), C("C", t, Sparse);
  TensorVar w("w", Type(type<double>(),{3}), Dense);

  auto spmm = forall(i,
                     forall(k,
                            where(forall(j, A(i,j) = w(j)),
                                  forall(j,   w(j) += B(i,k)*C(k,j))
                                  )
                            )
                     );
}

TEST(indexstmt, sddmm) {
  Type t(type<double>(), {3,3});
  TensorVar A("A", t, {Sparse, Dense});
  TensorVar B("B", t, {Sparse, Dense});
  TensorVar C("C", t, {Dense, Dense});
  TensorVar w("w", Type(type<double>(),{3}), Dense);

  // the below expression is the concrete index notation
  // where (consumer, producer)
  IndexStmt spmm = forall(i,
                     forall(k,
                            where(forall(j, A(i,j) = w(j)),
                                  forall(j,   w(j) += B(i,k)*C(k,j))
                                  )
                            )
                     );

  // after adding scheduling transformations to this concrete-topologically sorted index stmt
  //

  std::cout << spmm << std::endl;
  spmm = reorderLoopsTopologically(spmm);
  std::cout << "topologically reordered loops statement: " << spmm << std::endl;

  Kernel kernel = compile(spmm);

}


TEST(indexstmt, sddmmPlusSpmm) {

  // Y(i,l) = B(i,j)*C(i,k)*D(k,j) * F(j,l);
  // indexstmt order i, j, k, l
  //topologically reordered loops statement: forall(i, forall(k, forall(j, forall(l, Y(i,l) += B(i,j) * C(i,k) * D(k,j) * F(j,l), NotParallel, IgnoreRaces), NotParallel, IgnoreRaces), NotParallel, IgnoreRaces), NotParallel, IgnoreRaces)

  Type t(type<double>(), {3,3});
  TensorVar Y("Y", t, {Dense, Dense});
  TensorVar B("B", t, {Dense, Sparse});
  TensorVar C("C", t, {Dense, Dense});
  TensorVar D("D", t, {Dense, Dense});
  TensorVar E("E", t, {Dense, Dense});

  // TensorVar A("A", Type(type<double>(),{3}), );
  TensorVar A("A", Type());

  IndexStmt fused1 = 
  forall(i,
    forall(j,
      forall(k,
        forall(l, Y(i,l) += B(i,j) * C(i,k) * D(j,k) * E(j,l))
      )
    )
  );

  std::cout << "before topological sort" << fused1 << std::endl;
  fused1 = reorderLoopsTopologically(fused1);
  std::cout << "after topological sort" << fused1 << std::endl;

  Kernel kernel = compile(fused1);


  IndexStmt fused2 =
  forall(i,
    forall(j,
      where(
        forall(l, Y(i,l) += A * E(j,l)), // consumer
        forall(k, A += B(i,j)*C(i,k)*D(j,k)) // producer
      )
    )
  );

  Kernel kernel2 = compile(fused2);

} 

TEST(indexstmt, mttkrpPlusSpmm) {

  // ./bin/taco "A(i,m)=B(i,k,l)*C(k,j)*D(l,j)*E(j,m)" -f=A:dd:0,1 -f=B:sss:0,1,2 -f=C:dd:0,1 -f=D:dd:0,1 -f=E:dd:0,1

  // i = 11, k = 5, l = 7, j = 8;
  long unsigned int idim = 11, kdim = 5, ldim = 7, jdim = 8, mdim = 6;

  Type atype(type<double>(), {idim, mdim});
  Type btype(type<double>(), {idim, kdim, ldim});
  Type ctype(type<double>(), {kdim, jdim});
  Type dtype(type<double>(), {ldim, jdim});
  Type etype(type<double>(), {jdim, mdim});

  TensorVar A("A", atype, {Dense, Dense});
  TensorVar B("B", btype, {Sparse, Sparse, Sparse});
  TensorVar C("C", ctype, {Dense, Dense});
  TensorVar D("D", dtype, {Dense, Dense});
  TensorVar E("E", etype, {Dense, Dense});

  TensorVar ws("ws", Type(type<double>(), {jdim}) );

  IndexStmt fused1 = 
  forall(i,
    forall(k,
      forall(l,
        forall(j,
          forall(m, A(i,m) += B(i,k,l) * C(k,j) * D(l,j) * E(j,m))
        )
      )
    )
  );

  std::cout << "before topological sort" << fused1 << std::endl;
  fused1 = reorderLoopsTopologically(fused1);
  std::cout << "after topological sort" << fused1 << std::endl;

  Kernel kernel = compile(fused1);

  IndexStmt fused2 =
  forall(i,
    where(
      forall(j,
        forall(m, 
          A(i,m) += ws(j) * E(j,m)
        )
      )
      ,
      forall(k,
        forall(l,
          forall(j, 
            ws(j) += B(i,k,l) * C(k,j) * D(l,j)
          )
        )
      )
    )
  );

  Kernel kernel2 = compile(fused2);

}

// ./bin/taco "y(i)=A(i,j)*B(j,k)*v(k)" -f=y:d:0 -f=A:dd:0,1 -f=B:dd:0,1 -f=v:d:0
TEST(indexstmt, mmPlusSpmv) {

  //

  long unsigned int idim = 11, jdim = 8, kdim = 5;

  Type ytype(type<double>(), {idim});
  Type atype(type<double>(), {idim, jdim});
  Type btype(type<double>(), {jdim, kdim});
  Type vtype(type<double>(), {kdim});

  TensorVar y("y", ytype, {Dense});
  TensorVar A("A", atype, {Dense, Dense});
  TensorVar B("B", btype, {Dense, Dense});
  TensorVar v("v", vtype, {Dense});

  TensorVar ws("ws", Type(type<double>(), {jdim}) );

  IndexStmt fused1 = 
  forall(i,
    forall(j,
      forall(k,
        forall(m, y(i) += A(i,j) * B(j,k) * v(k))
      )
    )
  );

  std::cout << "before topological sort" << fused1 << std::endl;
  fused1 = reorderLoopsTopologically(fused1);
  std::cout << "after topological sort" << fused1 << std::endl;

  Kernel kernel = compile(fused1); 

  IndexStmt fused2 =
  where(
    forall(i,
      forall(j, 
        y(i) += A(i,j) * ws(j)
      )
    )
    ,
    forall(j,
      forall(k,
        ws(j) += B(j,k) * v(k)
      )
    )
  );

  Kernel kernel2 = compile(fused2);
}

