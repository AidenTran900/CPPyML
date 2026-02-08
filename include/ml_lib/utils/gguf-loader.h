#pragma once
#include "../math/matrix.h"
#include "../core/tokenizer.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <fstream>

enum class GGMLType : uint32_t {
    F32 = 0, F16 = 1,
    Q4_0 = 2, Q4_1 = 3,
    Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q8_1 = 9
};

enum class GGUFValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3,
    UINT32 = 4, INT32 = 5, FLOAT32 = 6, BOOL = 7,
    STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11,
    FLOAT64 = 12
};

struct GGUFValue {
    GGUFValueType type;
    uint64_t uint_val = 0;
    int64_t int_val = 0;
    double float_val = 0.0;
    bool bool_val = false;
    std::string string_val;
    std::vector<GGUFValue> array_val;
};

struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> dimensions;
    GGMLType type;
    uint64_t offset;
};

class GGUFFile {
    private:
        std::unordered_map<std::string, GGUFValue> metadata;
        std::vector<GGUFTensorInfo> tensors;
        std::string file_path;
        uint64_t tensor_data_offset;
        uint32_t alignment = 32;

        std::string readString(std::ifstream& f);
        GGUFValue readValue(std::ifstream& f, GGUFValueType type);
        void skipValue(std::ifstream& f, GGUFValueType type);
        static float f16ToF32(uint16_t h);

    public:
        bool load(const std::string& path);

        bool hasKey(const std::string& key) const;
        std::string getString(const std::string& key) const;
        uint64_t getUint(const std::string& key) const;
        int64_t getInt(const std::string& key) const;
        double getFloat(const std::string& key) const;
        const std::vector<GGUFValue>& getArray(const std::string& key) const;

        const std::vector<GGUFTensorInfo>& getTensors() const { return tensors; }
        const GGUFTensorInfo* findTensor(const std::string& name) const;

        std::string getArchitecture() const;
        int getNumLayers() const;
        int getEmbedDim() const;
        int getNumHeads() const;
        int getNumKVHeads() const;
        int getVocabSize() const;
        int getFFDim() const;
        int getContextLength() const;

        template<typename T>
        Matrix<T> loadTensor(const std::string& name) const;

        template<typename T>
        Matrix<T> loadTensor(const GGUFTensorInfo& info) const;

        void loadTokenizer(Tokenizer& tokenizer) const;
};
