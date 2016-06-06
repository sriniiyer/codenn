local Encoder, parent = torch.class('Encoder', 'nn.Module')

function Encoder:__init(config, params)
	parent.__init(self)
	self.emb_dim = config.emb_dim
	self.emb_out = config.emb_out
	self.params = params

  self.cell = self:new_cell(self.emb_out)

  self.paramx, self.paramdx = self.cell:getParameters()
  self.paramx:uniform(-self.params.init_weight, self.params.init_weight)
	self.paramdx:zero()

end

function Encoder:new_cell(dim)
	local x                = nn.Identity()()
  local h = nn.LookupTable(self.emb_dim, self.emb_out)(x)
  local transposedH = nn.Transpose({1, 2})(h)

  return  nn.gModule({x}, {transposedH}):cuda()
end

function Encoder:forward(batch)
  return self.cell:forward(batch.x)
end

function Encoder:backward(batch, sum_d_all_h)
  self.cell:backward(batch.x, sum_d_all_h)
end

function Encoder:updateParameters()
  update_params(self.paramx, self.paramdx, self.params.max_grad_norm, self.params.learningRate, self.params.normalize)
	self.paramdx:zero()
end
