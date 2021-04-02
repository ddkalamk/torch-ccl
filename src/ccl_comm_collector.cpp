
#include <sys/types.h>
#include <unistd.h>
#include "ccl_comm_collector.h"
#include "utils.h"


namespace torch_ccl {

ccl::shared_ptr_class<ccl::kvs> CCLMember::get_kvs(int rank, c10d::Store& store) {
  if (kvs)
    return kvs;
  // Each process group is with different store, so we use the unique key for
  // broadcast the bootstrap network information.
  std::string storeKey = "ccl_kvs";

  // Rank 0 broadcast the bootstrap network information to other ranks
  if (rank == 0) {
    call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
        kvs = ccl::create_main_kvs();
    });
    ccl::kvs::address_type main_addr = kvs->get_address();
    auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
    store.set(storeKey, ccl_kvs_addr);
  }
  else {
    auto ccl_kvs_addr = store.get(storeKey);
    if (ccl_kvs_addr.size() != ccl::kvs::address_max_size) {
      throw std::runtime_error(
              "Unexpected ccl kvs addr from the store\n");
    }
    ccl::kvs::address_type main_addr;
    std::copy_n(std::make_move_iterator(ccl_kvs_addr.begin()),
                ccl::kvs::address_max_size,
                main_addr.begin());
    call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
        kvs = ccl::create_kvs(main_addr);
    });
  }

  return kvs;
}

}