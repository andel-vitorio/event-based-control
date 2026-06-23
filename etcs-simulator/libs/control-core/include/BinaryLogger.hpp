#include <fstream>
#include <vector>
#include <string>
#include <mutex>
#include <type_traits>
#include <stdexcept>
#include <filesystem> // Adicione este include

class BinaryLogger
{
private:
  std::ofstream file_;
  std::mutex mutex_;

public:
  // Agora aceita fs::path
  explicit BinaryLogger(const std::filesystem::path &filename)
  {
    file_.open(filename, std::ios::binary | std::ios::out);
    if (!file_.is_open())
      throw std::runtime_error("Cannot open file: " + filename.string());
  }

  template <typename T>
  void log(const std::vector<T> &data)
  {
    static_assert(std::is_trivially_copyable<T>::value, "Type must be trivially copyable");

    std::lock_guard<std::mutex> lock(mutex_);
    file_.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T));
  }

  // Agora aceita fs::path
  template <typename T>
  static void dump(const std::filesystem::path &filename, const std::vector<T> &data)
  {
    static_assert(std::is_trivially_copyable<T>::value, "Type must be trivially copyable");

    std::ofstream ofs(filename, std::ios::binary | std::ios::out);
    if (!ofs)
      throw std::runtime_error("Cannot open file for dumping: " + filename.string());

    ofs.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T));
  }
};