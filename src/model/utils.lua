-- Adapted from https://github.com/wojciechz/learning_to_execute

--[[
--
  Copyright 2014 Google Inc. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
]]--

require "cutorch"
require "cunn"
require "nn"
require 'nngraph'

gModule = torch.getmetatable('nn.gModule')

function clone (t) -- deep-copy a table
    if type(t) ~= "table" then return t end
    local meta = getmetatable(t)
    local target = {}
    for k, v in pairs(t) do
        if type(v) == "table" then
            target[k] = clone(v)
        else
            target[k] = v
        end
    end
    setmetatable(target, meta)
    return target
end

function table.shallow_copy(t)
  local t2 = {}
  for k,v in pairs(t) do
    t2[k] = v
  end
  return t2
end

function shuffleTable(t)
    local rand = math.random 
    assert(t, "shuffleTable() expected a table, got nil")
    local iterations = #t
    local j
    
    for i = iterations, 2, -1 do
        j = rand(i)
        t[i], t[j] = t[j], t[i]
    end
end

function lstm(i, prev_c, prev_h, inp_dim)
	function new_input_sum()
		local i2h            = nn.Linear(inp_dim, params.rnn_size)
		local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
		return nn.CAddTable()({i2h(i), h2h(prev_h)})
	end
	local in_gate          = nn.Sigmoid()(new_input_sum()):annotate{name = 'in_gate'}
	local forget_gate      = nn.Sigmoid()(new_input_sum()):annotate{name = 'forget_gate'}
	local in_gate2         = nn.Tanh()(new_input_sum()):annotate{name = 'in_gate2'}
	local next_c           = nn.CAddTable()({
		nn.CMulTable()({forget_gate, prev_c}),
		nn.CMulTable()({in_gate,     in_gate2})
	})
	local out_gate         = nn.Sigmoid()(new_input_sum()):annotate{name = 'out_gate'}
	local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
	return next_c, next_h
end

function reset_state(state)
	state.acc = 0
	state.count = 0
	state.normal = 0
end


function g_disable_dropout_all(nets)
	for i = 1, #nets do
		g_disable_dropout(nets[i])
	end
end

function g_enable_dropout_all(nets, p)
	for i = 1, #nets do
		g_enable_dropout(nets[i], p)
	end
end

function g_disable_dropout(node)
	node:evaluate()
end

function g_enable_dropout(node, p)
	node:training()
end

--[[ Creates clones of the given network.
The clones share all weights and gradWeights with the original network.
Accumulating of the gradients sums the gradients properly.
The clone also allows parameters for which gradients are never computed
to be shared. Such parameters must be returns by the parametersNoGrad
method, which can be null.
--]]
function cloneManyTimes(net, T)
  local clones = {}
  local params, gradParams = net:parameters()
  if params == nil then
    params = {}
  end
  local paramsNoGrad
  if net.parametersNoGrad then
    paramsNoGrad = net:parametersNoGrad()
  end
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    local cloneParamsNoGrad
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    if paramsNoGrad then
      cloneParamsNoGrad = clone:parametersNoGrad()
      for i =1,#paramsNoGrad do
        cloneParamsNoGrad[i]:set(paramsNoGrad[i])
      end
    end
    clones[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end

function getInCuda(batch)
		newBatch = {}
		newBatch.x = batch.x:cuda()
		newBatch.y = batch.y:cuda()
		newBatch.fmask = batch.fmask:cuda()
		newBatch.infmask = newBatch.fmask:clone():add(-1):mul(1000000000)
		newBatch.mask = batch.mask:cuda()
		newBatch.xsizes = batch.xsizes:cuda()
		newBatch.maxX = batch.maxX
		newBatch.maxY = batch.maxY
		newBatch.xsizes = batch.xsizes:cuda()
		newBatch.ids = batch.ids:clone()
		newBatch.code= clone(batch.code)
		return newBatch
end

function str_hash(str)
  local hash = 1
  for i = 1, #str, 2 do
    hash = math.fmod(hash * 12345, 452930459) +
    ((string.byte(str, i) or (len - i)) * 67890) +
    ((string.byte(str, i + 1) or i) * 13579)
  end
  return hash
end

function init_gpu(gpuidx)
  cutorch.setDevice(gpuidx)
  make_deterministic(1)
end

function make_deterministic(seed)
  torch.manualSeed(seed)
  cutorch.manualSeed(seed)
  torch.zeros(1, 1):cuda():uniform()
end

function copy_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:copy(from[i])
  end
end

function os.capture(cmd)
  local f = assert(io.popen(cmd, 'r'))
  local s = assert(f:read('*a'))
  f:close()
  s = string.gsub(s, '[\n\r]+', ' ')
  return s
end

function argmax(vector)
  if vector:dim() == 1 then
    for i = 1, vector:size(1) do
      if vector[i] == vector:max() then
        return i
      end
    end
  else
    error("Argmax only supports vectors")
  end
end

function script_path()
  return debug.getinfo(2, "S").source:sub(2)
end

floor = torch.floor
ceil = torch.ceil
random = torch.random

function update_params(px, pdx, mx, lr, donorm)
  local norm = pdx:norm()

  if donorm == 1 and norm > mx then
    local shrink_factor = mx / norm
    pdx:mul(shrink_factor)
  end
  px:add(pdx:mul(-lr))
end

