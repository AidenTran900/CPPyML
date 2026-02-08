#include "ml_lib/utils/gguf-loader.h"
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <sstream>

// F16 → F32 conversion (IEEE 754 half-precision)
float GGUFFile::f16ToF32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    uint32_t result;
    if (exp == 0) {
        if (mant == 0) {
            result = sign << 31;
        } else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= ~0x400;
            result = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        result = (sign << 31) | (0xff << 23) | (mant << 13);
    } else {
        result = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }

    float f;
    std::memcpy(&f, &result, 4);
    return f;
}

std::string GGUFFile::readString(std::ifstream& f) {
    uint64_t len;
    f.read(reinterpret_cast<char*>(&len), 8);
    std::string s(len, '\0');
    f.read(&s[0], len);
    return s;
}

GGUFValue GGUFFile::readValue(std::ifstream& f, GGUFValueType type) {
    GGUFValue val;
    val.type = type;

    switch (type) {
        case GGUFValueType::UINT8: {
            uint8_t v; f.read(reinterpret_cast<char*>(&v), 1);
            val.uint_val = v; break;
        }
        case GGUFValueType::INT8: {
            int8_t v; f.read(reinterpret_cast<char*>(&v), 1);
            val.int_val = v; break;
        }
        case GGUFValueType::UINT16: {
            uint16_t v; f.read(reinterpret_cast<char*>(&v), 2);
            val.uint_val = v; break;
        }
        case GGUFValueType::INT16: {
            int16_t v; f.read(reinterpret_cast<char*>(&v), 2);
            val.int_val = v; break;
        }
        case GGUFValueType::UINT32: {
            uint32_t v; f.read(reinterpret_cast<char*>(&v), 4);
            val.uint_val = v; break;
        }
        case GGUFValueType::INT32: {
            int32_t v; f.read(reinterpret_cast<char*>(&v), 4);
            val.int_val = v; break;
        }
        case GGUFValueType::FLOAT32: {
            float v; f.read(reinterpret_cast<char*>(&v), 4);
            val.float_val = v; break;
        }
        case GGUFValueType::BOOL: {
            uint8_t v; f.read(reinterpret_cast<char*>(&v), 1);
            val.bool_val = (v != 0); break;
        }
        case GGUFValueType::STRING: {
            val.string_val = readString(f); break;
        }
        case GGUFValueType::ARRAY: {
            uint32_t elem_type;
            f.read(reinterpret_cast<char*>(&elem_type), 4);
            uint64_t count;
            f.read(reinterpret_cast<char*>(&count), 8);
            val.array_val.reserve(count);
            for (uint64_t i = 0; i < count; i++) {
                val.array_val.push_back(readValue(f, static_cast<GGUFValueType>(elem_type)));
            }
            break;
        }
        case GGUFValueType::UINT64: {
            uint64_t v; f.read(reinterpret_cast<char*>(&v), 8);
            val.uint_val = v; break;
        }
        case GGUFValueType::INT64: {
            int64_t v; f.read(reinterpret_cast<char*>(&v), 8);
            val.int_val = v; break;
        }
        case GGUFValueType::FLOAT64: {
            double v; f.read(reinterpret_cast<char*>(&v), 8);
            val.float_val = v; break;
        }
    }

    return val;
}

bool GGUFFile::load(const std::string& path) {
    file_path = path;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    // Read magic
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x46554747) return false; // "GGUF" little-endian

    // Read version
    uint32_t version;
    f.read(reinterpret_cast<char*>(&version), 4);
    if (version < 2) return false;

    // Read counts
    uint64_t tensor_count, metadata_kv_count;
    f.read(reinterpret_cast<char*>(&tensor_count), 8);
    f.read(reinterpret_cast<char*>(&metadata_kv_count), 8);

    // Read metadata key-value pairs
    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        std::string key = readString(f);
        uint32_t value_type;
        f.read(reinterpret_cast<char*>(&value_type), 4);
        GGUFValue value = readValue(f, static_cast<GGUFValueType>(value_type));
        metadata[key] = std::move(value);
    }

    // Check for custom alignment
    if (hasKey("general.alignment")) {
        alignment = static_cast<uint32_t>(getUint("general.alignment"));
    }

    // Read tensor info entries
    tensors.resize(tensor_count);
    for (uint64_t i = 0; i < tensor_count; i++) {
        tensors[i].name = readString(f);

        uint32_t n_dims;
        f.read(reinterpret_cast<char*>(&n_dims), 4);

        tensors[i].dimensions.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; d++) {
            f.read(reinterpret_cast<char*>(&tensors[i].dimensions[d]), 8);
        }

        uint32_t type;
        f.read(reinterpret_cast<char*>(&type), 4);
        tensors[i].type = static_cast<GGMLType>(type);

        f.read(reinterpret_cast<char*>(&tensors[i].offset), 8);
    }

    // Calculate tensor data start (align current position)
    uint64_t pos = f.tellg();
    tensor_data_offset = ((pos + alignment - 1) / alignment) * alignment;

    return true;
}

// Metadata accessors

bool GGUFFile::hasKey(const std::string& key) const {
    return metadata.find(key) != metadata.end();
}

std::string GGUFFile::getString(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) return "";
    return it->second.string_val;
}

uint64_t GGUFFile::getUint(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) return 0;
    const auto& v = it->second;
    if (v.type == GGUFValueType::UINT32 || v.type == GGUFValueType::UINT64 ||
        v.type == GGUFValueType::UINT16 || v.type == GGUFValueType::UINT8)
        return v.uint_val;
    return static_cast<uint64_t>(v.int_val);
}

int64_t GGUFFile::getInt(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) return 0;
    const auto& v = it->second;
    if (v.type == GGUFValueType::INT32 || v.type == GGUFValueType::INT64 ||
        v.type == GGUFValueType::INT16 || v.type == GGUFValueType::INT8)
        return v.int_val;
    return static_cast<int64_t>(v.uint_val);
}

double GGUFFile::getFloat(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) return 0.0;
    return it->second.float_val;
}

const std::vector<GGUFValue>& GGUFFile::getArray(const std::string& key) const {
    static const std::vector<GGUFValue> empty;
    auto it = metadata.find(key);
    if (it == metadata.end()) return empty;
    return it->second.array_val;
}

const GGUFTensorInfo* GGUFFile::findTensor(const std::string& name) const {
    for (const auto& t : tensors) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

// Architecture metadata helpers

std::string GGUFFile::getArchitecture() const {
    return getString("general.architecture");
}

int GGUFFile::getNumLayers() const {
    std::string arch = getArchitecture();
    return static_cast<int>(getUint(arch + ".block_count"));
}

int GGUFFile::getEmbedDim() const {
    std::string arch = getArchitecture();
    return static_cast<int>(getUint(arch + ".embedding_length"));
}

int GGUFFile::getNumHeads() const {
    std::string arch = getArchitecture();
    return static_cast<int>(getUint(arch + ".attention.head_count"));
}

int GGUFFile::getNumKVHeads() const {
    std::string arch = getArchitecture();
    std::string key = arch + ".attention.head_count_kv";
    if (hasKey(key)) return static_cast<int>(getUint(key));
    return getNumHeads();
}

int GGUFFile::getVocabSize() const {
    const auto& tokens = getArray("tokenizer.ggml.tokens");
    return static_cast<int>(tokens.size());
}

int GGUFFile::getFFDim() const {
    std::string arch = getArchitecture();
    return static_cast<int>(getUint(arch + ".feed_forward_length"));
}

int GGUFFile::getContextLength() const {
    std::string arch = getArchitecture();
    return static_cast<int>(getUint(arch + ".context_length"));
}

// Tensor loading

template<typename T>
Matrix<T> GGUFFile::loadTensor(const std::string& name) const {
    const GGUFTensorInfo* info = findTensor(name);
    if (!info) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return loadTensor<T>(*info);
}

template<typename T>
Matrix<T> GGUFFile::loadTensor(const GGUFTensorInfo& info) const {
    // Determine matrix dimensions: treat as 2D (rows x cols)
    int rows = 1, cols = 1;
    if (info.dimensions.size() >= 1) cols = static_cast<int>(info.dimensions[0]);
    if (info.dimensions.size() >= 2) rows = static_cast<int>(info.dimensions[1]);

    Matrix<T> mat(rows, cols);

    std::ifstream f(file_path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot reopen GGUF file: " + file_path);
    }

    f.seekg(tensor_data_offset + info.offset);

    uint64_t num_elements = static_cast<uint64_t>(rows) * cols;

    if (info.type == GGMLType::F32) {
        std::vector<float> buf(num_elements);
        f.read(reinterpret_cast<char*>(buf.data()), num_elements * 4);
        for (uint64_t i = 0; i < num_elements; i++) {
            mat(static_cast<int>(i / cols), static_cast<int>(i % cols)) = static_cast<T>(buf[i]);
        }
    } else if (info.type == GGMLType::F16) {
        std::vector<uint16_t> buf(num_elements);
        f.read(reinterpret_cast<char*>(buf.data()), num_elements * 2);
        for (uint64_t i = 0; i < num_elements; i++) {
            mat(static_cast<int>(i / cols), static_cast<int>(i % cols)) = static_cast<T>(f16ToF32(buf[i]));
        }
    } else {
        std::ostringstream oss;
        oss << "Unsupported tensor type " << static_cast<int>(info.type)
            << " for tensor '" << info.name << "'. Only F32 and F16 are supported. "
            << "Use a non-quantized GGUF model (e.g. F16 or F32).";
        throw std::runtime_error(oss.str());
    }

    return mat;
}

// Tokenizer loading from GGUF metadata

void GGUFFile::loadTokenizer(Tokenizer& tokenizer) const {
    tokenizer.setType(TOKENIZER_TYPE::BPE);

    // Load vocab: token at index i → id i
    const auto& tokens = getArray("tokenizer.ggml.tokens");
    std::unordered_map<std::string, int> vocab;
    for (size_t i = 0; i < tokens.size(); i++) {
        vocab[tokens[i].string_val] = static_cast<int>(i);
    }
    tokenizer.setVocab(vocab);

    // Load BPE merges if available
    if (hasKey("tokenizer.ggml.merges")) {
        const auto& merges = getArray("tokenizer.ggml.merges");
        std::vector<std::pair<std::string, std::string>> merge_pairs;
        merge_pairs.reserve(merges.size());

        for (const auto& m : merges) {
            const std::string& merge_str = m.string_val;
            size_t space_pos = merge_str.find(' ');
            if (space_pos != std::string::npos) {
                merge_pairs.push_back({
                    merge_str.substr(0, space_pos),
                    merge_str.substr(space_pos + 1)
                });
            }
        }
        tokenizer.setMerges(merge_pairs);
    }
}

// Explicit template instantiations
template Matrix<float> GGUFFile::loadTensor<float>(const std::string& name) const;
template Matrix<double> GGUFFile::loadTensor<double>(const std::string& name) const;
template Matrix<float> GGUFFile::loadTensor<float>(const GGUFTensorInfo& info) const;
template Matrix<double> GGUFFile::loadTensor<double>(const GGUFTensorInfo& info) const;
